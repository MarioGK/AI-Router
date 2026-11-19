package dashboard

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"mime"
	"net/http"
	"path"
	"time"
)

// static holds the built stylesheet (make css) and the vendored Datastar
// bundle (make datastar).
//
//go:embed static
var static embed.FS

// asset is an embedded file with the content hash that busts caches.
type asset struct {
	body []byte
	hash string
}

var assets = func() map[string]asset {
	entries, err := static.ReadDir("static")
	if err != nil {
		panic(err)
	}
	loaded := map[string]asset{}
	for _, entry := range entries {
		body, err := static.ReadFile("static/" + entry.Name())
		if err != nil {
			panic(err)
		}
		sum := sha256.Sum256(body)
		loaded[entry.Name()] = asset{body: body, hash: hex.EncodeToString(sum[:6])}
	}
	return loaded
}()

// assetURL is the versioned URL of an embedded file.
func assetURL(name string) string {
	return "/dashboard/assets/" + name + "?v=" + assets[name].hash
}

// serveAsset serves an embedded file; a request for the current version
// may be cached forever.
func serveAsset(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	file, ok := assets[name]
	if !ok {
		http.NotFound(w, r)
		return
	}
	header := w.Header()
	header.Set("Content-Type", mime.TypeByExtension(path.Ext(name)))
	header.Set("ETag", `"`+file.hash+`"`)
	if r.URL.Query().Get("v") == file.hash {
		header.Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		header.Set("Cache-Control", "no-cache")
	}
	http.ServeContent(w, r, name, time.Time{}, bytes.NewReader(file.body))
}
