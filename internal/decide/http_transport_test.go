package decide

import (
	"net/http"
	"net/http/httptest"
)

// testEndpoint exercises HTTP request and response contracts without a socket.
type testEndpoint struct {
	URL    string
	client *http.Client
}

func newTestEndpoint(handler http.Handler) *testEndpoint {
	return &testEndpoint{
		URL: "https://decider.invalid",
		client: &http.Client{Transport: judgeMetadataTransport(func(r *http.Request) (*http.Response, error) {
			defer r.Body.Close()
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, r)
			return recorder.Result(), nil
		})},
	}
}

func (s *testEndpoint) Client() *http.Client { return s.client }
