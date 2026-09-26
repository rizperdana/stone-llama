package cli

import (
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

// liveCollectionJSON is an unmodified capture of
// GET https://huggingface.co/api/collections/prism-ml/bonsai-2-6aab292b9fa72a78652429df
// (2026-09-26). The real payload is FLAT: each item carries top-level
// {"id": …, "type": "model"|"space"}, never a nested "item" object —
// the nested-only parser returned "collection … has no models" for it.
const liveCollectionJSON = `{"slug":"prism-ml/bonsai-2-6aab292b9fa72a78652429df","title":"Bonsai-2","description":"","lastUpdated":"2026-09-17T21:06:21.356Z","gating":false,"owner":{"_id":"6920b7de32b82aa818ac67cd","avatarUrl":"https://cdn-avatars.huggingface.co/v1/production/uploads/6920b71747acb07530915d41/mEaEo0tgAYZn-NB6S-nuP.png","fullname":"Prism ML","name":"prism-ml","type":"org","isHf":false,"isHfAdmin":false,"isMod":false,"plan":"team","followerCount":3873,"isUserFollowing":false},"items":[{"author":"webml-community","authorData":{"_id":"65ef8f2d9540f72aff05a3c4","avatarUrl":"https://cdn-avatars.huggingface.co/v1/production/uploads/61b253b7ac5ecaae3d1efe0c/UJbVX1QgBUe21A8nm5zWL.png","fullname":"WebML Community","name":"webml-community","type":"org","isHf":false,"isHfAdmin":false,"isMod":false,"followerCount":1428,"isUserFollowing":false},"colorFrom":"pink","colorTo":"blue","createdAt":"2026-09-17T14:46:17.000Z","disabled":false,"emoji":"🌳","id":"webml-community/ternary-bonsai-2-webgpu-kernels","lastModified":"2026-09-17T18:25:15.000Z","likes":136,"pinned":false,"private":false,"sdk":"static","repoType":"space","runtime":{"stage":"RUNNING","hardware":{"current":null,"requested":null},"replicas":{"requested":1,"current":1}},"shortDescription":"Run Ternary-Bonsai-2-27B locally in your browser on WebGPU","title":"Ternary Bonsai 2 WebGPU Kernels","isLikedByUser":false,"ai_short_description":"Chat with a 27B ternary AI model in your browser","ai_category":"Chatbots","tags":["static","region:us"],"featured":true,"visibility":"public","type":"space","position":0,"_id":"6aac564911ba981a1b9ae180"},{"author":"prism-ml","authorData":{"_id":"6920b7de32b82aa818ac67cd","avatarUrl":"https://cdn-avatars.huggingface.co/v1/production/uploads/6920b71747acb07530915d41/mEaEo0tgAYZn-NB6S-nuP.png","fullname":"Prism ML","name":"prism-ml","type":"org","isHf":false,"isHfAdmin":false,"isMod":false,"plan":"team","followerCount":3873,"isUserFollowing":false},"downloads":3109078,"gated":false,"id":"prism-ml/Ternary-Bonsai-2-27B-gguf","availableInferenceProviders":[],"lastModified":"2026-09-25T21:58:06.000Z","likes":2099,"pipeline_tag":"text-generation","private":false,"repoType":"model","isLikedByUser":false,"widgetOutputUrls":[],"numParameters":26895998464,"type":"model","position":1,"_id":"6aab295383f28b204f9f73fc"},{"author":"prism-ml","authorData":{"_id":"6920b7de32b82aa818ac67cd","avatarUrl":"https://cdn-avatars.huggingface.co/v1/production/uploads/6920b71747acb07530915d41/mEaEo0tgAYZn-NB6S-nuP.png","fullname":"Prism ML","name":"prism-ml","type":"org","isHf":false,"isHfAdmin":false,"isMod":false,"plan":"team","followerCount":3873,"isUserFollowing":false},"downloads":54097,"gated":false,"id":"prism-ml/Ternary-Bonsai-2-27B-mlx-2bit","availableInferenceProviders":[],"lastModified":"2026-09-22T21:18:49.000Z","likes":378,"pipeline_tag":"text-generation","private":false,"repoType":"model","isLikedByUser":false,"widgetOutputUrls":[],"numParameters":27359638768,"type":"model","position":2,"_id":"6aab29ac6529d032bf4cc14c"},{"author":"prism-ml","authorData":{"_id":"6920b7de32b82aa818ac67cd","avatarUrl":"https://cdn-avatars.huggingface.co/v1/production/uploads/6920b71747acb07530915d41/mEaEo0tgAYZn-NB6S-nuP.png","fullname":"Prism ML","name":"prism-ml","type":"org","isHf":false,"isHfAdmin":false,"isMod":false,"plan":"team","followerCount":3873,"isUserFollowing":false},"downloads":6243,"gated":false,"id":"prism-ml/Ternary-Bonsai-2-27B-gguf-dev","availableInferenceProviders":[],"lastModified":"2026-09-17T18:44:02.000Z","likes":31,"pipeline_tag":"text-generation","private":false,"repoType":"model","isLikedByUser":false,"widgetOutputUrls":[],"numParameters":26895998464,"type":"model","position":3,"_id":"6aaba5f30e01b865075f135c"}],"position":0,"theme":"pink","private":false,"shareUrl":"https://hf.co/collections/prism-ml/bonsai-2","upvotes":149,"isUpvotedByUser":false}`

// TestCollectionIDsLiveFlatPayload fails on the nested-only parser:
// the live API's flat items all decode as empty and the collection is
// reported as having no models.
func TestCollectionIDsLiveFlatPayload(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/collections/prism-ml/bonsai-2-6aab292b9fa72a78652429df" {
			http.NotFound(w, r)
			return
		}
		io.WriteString(w, liveCollectionJSON)
	}))
	defer ts.Close()

	ids, err := collectionIDs(ts.URL, "", "prism-ml/bonsai-2-6aab292b9fa72a78652429df")
	if err != nil {
		t.Fatalf("collectionIDs: %v (flat payload must parse)", err)
	}
	want := []string{
		"prism-ml/Ternary-Bonsai-2-27B-gguf",
		"prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
		"prism-ml/Ternary-Bonsai-2-27B-gguf-dev",
	}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("ids = %v, want %v (models only; the space item is filtered)", ids, want)
	}
}

// TestCollectionIDsNestedItemStillAccepted keeps the older nested
// {"item": {"id": …}} shape working — handle both, assume neither.
func TestCollectionIDsNestedItemStillAccepted(t *testing.T) {
	const nested = `{"items":[{"item":{"id":"org/nested-model","type":"model"}},{"id":"org/flat-model","type":"model"},{"id":"org/a-space","type":"space"}],"next":null}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, nested)
	}))
	defer ts.Close()

	ids, err := collectionIDs(ts.URL, "", "org/collection")
	if err != nil {
		t.Fatalf("collectionIDs: %v", err)
	}
	want := []string{"org/nested-model", "org/flat-model"}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("ids = %v, want %v", ids, want)
	}
}
