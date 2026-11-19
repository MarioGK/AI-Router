package route

import "runtime/debug"

// version is the router build recorded with each decision: the VCS
// revision, marked when the tree was modified.
var version = func() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return ""
	}
	revision, modified := "", false
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value[:min(len(setting.Value), 12)]
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}
	if modified {
		revision += "+dirty"
	}
	return revision
}()
