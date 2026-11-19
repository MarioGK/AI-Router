// Package proxy forwards requests to CLIProxy faithfully: every end-to-end
// header (User-Agent, originator, anthropic-beta, session ids, auth) and
// every body byte except a routed model value reach it unchanged, and
// responses stream back chunk by chunk.
package proxy

import (
	"bytes"
	"context"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Forwarder sends buffered requests to one upstream.
type Forwarder struct {
	base   *url.URL
	client *http.Client
}

// NewClient returns the HTTP client for CLIProxy calls: pooled keep-alive
// connections, no overall timeout (responses stream as long as they need),
// no transparent compression and no redirect following.
func NewClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			MaxIdleConns:          128,
			MaxIdleConnsPerHost:   64,
			IdleConnTimeout:       90 * time.Second,
			ExpectContinueTimeout: time.Second,
			// The client's own Accept-Encoding passes through and bodies are
			// relayed as they are.
			DisableCompression: true,
		},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}

// New returns a forwarder for the upstream base URL.
func New(base *url.URL, client *http.Client) *Forwarder {
	return &Forwarder{base: base, client: client}
}

// Send forwards the incoming request with the given body. There is no
// overall timeout: model responses stream for as long as they need, and a
// client disconnect cancels ctx.
func (f *Forwarder) Send(ctx context.Context, in *http.Request, body []byte) (*http.Response, error) {
	target := *f.base
	target.Path = strings.TrimSuffix(f.base.Path, "/") + in.URL.Path
	target.RawPath = ""
	target.RawQuery = in.URL.RawQuery
	out, err := http.NewRequestWithContext(ctx, in.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	out.Header = endToEnd(in.Header)
	for _, name := range private {
		out.Header.Del(name)
	}
	if len(body) == 0 {
		out.Body = http.NoBody
	}
	out.ContentLength = int64(len(body))
	return f.client.Do(out)
}

// Copy writes an upstream response to the client, flushing after every
// read so server-sent events are not held back. Every chunk is written to
// tap (the usage meter, the watchdog) as it arrives from the upstream, before
// the client, so a slow client is never taken for a silent upstream.
func Copy(w http.ResponseWriter, response *http.Response, tap io.Writer) error {
	defer response.Body.Close()
	header := w.Header()
	for name, values := range endToEnd(response.Header) {
		header[name] = values
	}
	w.WriteHeader(response.StatusCode)
	controller := http.NewResponseController(w)
	buffer := make([]byte, 32<<10)
	for {
		n, err := response.Body.Read(buffer)
		if n > 0 {
			if tap != nil {
				_, _ = tap.Write(buffer[:n])
			}
			if _, writeErr := w.Write(buffer[:n]); writeErr != nil {
				return writeErr
			}
			_ = controller.Flush()
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// Discard drains and closes a response that will not be sent to the client.
func Discard(response *http.Response) {
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 1<<20))
	response.Body.Close()
}

// private headers never reach CLIProxy from a client: the management key
// is the router's own, and the forwarding headers would let a client pose
// as another address to CLIProxy's access checks.
var private = []string{"Expect", "X-Management-Key", "X-Forwarded-For", "X-Real-Ip", "Forwarded"}

// hopByHop headers describe one connection and are never forwarded.
var hopByHop = []string{
	"Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
	"Proxy-Connection", "Te", "Trailer", "Transfer-Encoding", "Upgrade",
}

func endToEnd(in http.Header) http.Header {
	out := in.Clone()
	for _, field := range in.Values("Connection") {
		for _, name := range strings.Split(field, ",") {
			out.Del(strings.TrimSpace(name))
		}
	}
	for _, name := range hopByHop {
		out.Del(name)
	}
	return out
}
