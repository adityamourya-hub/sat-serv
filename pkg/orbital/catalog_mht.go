package orbital

import (
	"math"
	"sort"
	"time"

	safemath "ssis/pkg/math"
)

type CatalogObjectEstimate struct {
	ObjectID   int
	Name       string
	State      State
	Covariance [][]float64
	LastUpdate int64
}

type AssociationObservation struct {
	TrackID       int
	TimestampUnix int64
	State         State
	Covariance    [][]float64
}

type MHTHypothesis struct {
	ObjectID      int
	Variant       string
	Posterior     float64
	SupportTracks int
	LastUpdate    int64
}

type MHTDecision struct {
	TrackID       int
	WinnerObject  int
	WinnerVariant string
	Posterior     float64
	UCT           bool
	Promoted      bool
	Hypotheses    []MHTHypothesis
}

type MHTLite struct {
	TrackHypotheses map[int][]MHTHypothesis
	ManeuverFlags   map[int]bool
	PromoteTracks   int
}

func NewMHTLite() *MHTLite {
	return &MHTLite{
		TrackHypotheses: make(map[int][]MHTHypothesis),
		ManeuverFlags:   make(map[int]bool),
		PromoteTracks:   3,
	}
}

func (m *MHTLite) FlagManeuver(objectID int) {
	m.ManeuverFlags[objectID] = true
}

func (m *MHTLite) Associate(obs AssociationObservation, objects []CatalogObjectEstimate, gateMahalanobis float64, maxHyp int) MHTDecision {
	if gateMahalanobis <= 0 {
		gateMahalanobis = 6.0
	}
	if maxHyp <= 0 {
		maxHyp = 6
	}
	if obs.TimestampUnix == 0 {
		obs.TimestampUnix = time.Now().Unix()
	}

	h := append([]MHTHypothesis(nil), m.TrackHypotheses[obs.TrackID]...)
	cands := make([]MHTHypothesis, 0)
	for _, obj := range objects {
		mahal := stateMahalanobis(obs.State, obs.Covariance, obj.State, obj.Covariance)
		if mahal > gateMahalanobis {
			continue
		}
		ll := -0.5 * mahal * mahal
		if m.ManeuverFlags[obj.ObjectID] {
			cands = append(cands, MHTHypothesis{ObjectID: obj.ObjectID, Variant: "legacy_orbit", Posterior: math.Exp(ll - 0.2), SupportTracks: 1, LastUpdate: obs.TimestampUnix})
			cands = append(cands, MHTHypothesis{ObjectID: obj.ObjectID, Variant: "new_orbit", Posterior: math.Exp(ll + 0.2), SupportTracks: 1, LastUpdate: obs.TimestampUnix})
		} else {
			cands = append(cands, MHTHypothesis{ObjectID: obj.ObjectID, Variant: "nominal", Posterior: math.Exp(ll), SupportTracks: 1, LastUpdate: obs.TimestampUnix})
		}
	}

	if len(cands) == 0 {
		h = mergeHypothesis(h, MHTHypothesis{ObjectID: 0, Variant: "uct", Posterior: 1, SupportTracks: 1, LastUpdate: obs.TimestampUnix})
		h = normalizeHyp(h)
		h = pruneHyp(h, maxHyp)
		m.TrackHypotheses[obs.TrackID] = h
		w := winnerHyp(h)
		return MHTDecision{TrackID: obs.TrackID, UCT: true, WinnerObject: w.ObjectID, WinnerVariant: w.Variant, Posterior: w.Posterior, Hypotheses: h}
	}

	for _, c := range cands {
		h = mergeHypothesis(h, c)
	}
	h = normalizeHyp(h)
	w := winnerHyp(h)
	promoted := false
	if w.ObjectID != 0 && w.Variant == "new_orbit" && w.SupportTracks >= m.PromoteTracks && w.Posterior >= 0.55 {
		promoted = true
		delete(m.ManeuverFlags, w.ObjectID)
	}
	h = pruneHyp(h, maxHyp)
	m.TrackHypotheses[obs.TrackID] = h

	uct := w.ObjectID == 0 || w.Posterior < 0.35
	return MHTDecision{TrackID: obs.TrackID, WinnerObject: w.ObjectID, WinnerVariant: w.Variant, Posterior: w.Posterior, UCT: uct, Promoted: promoted, Hypotheses: h}
}

func stateMahalanobis(obs State, obsCov [][]float64, obj State, objCov [][]float64) float64 {
	d := []float64{obs.X - obj.X, obs.Y - obj.Y, obs.Z - obj.Z}
	cov := safemath.ZeroMatrix(3, 3)
	for i := 0; i < 3; i++ {
		cov[i][i] = 2.0
		if len(obsCov) > i && len(obsCov[i]) > i {
			cov[i][i] += obsCov[i][i]
		}
		if len(objCov) > i && len(objCov[i]) > i {
			cov[i][i] += objCov[i][i]
		}
	}
	dist, _, err := safemath.MahalanobisDistance(d, cov, safemath.DefaultRegularizationEpsilon)
	if err != nil {
		return 1e9
	}
	if math.IsNaN(dist) || math.IsInf(dist, 0) {
		return 1e9
	}
	return dist
}

func mergeHypothesis(h []MHTHypothesis, in MHTHypothesis) []MHTHypothesis {
	for i := range h {
		if h[i].ObjectID == in.ObjectID && h[i].Variant == in.Variant {
			h[i].Posterior = 0.4*h[i].Posterior + 0.6*(h[i].Posterior+in.Posterior)
			h[i].SupportTracks++
			h[i].LastUpdate = in.LastUpdate
			return h
		}
	}
	return append(h, in)
}

func normalizeHyp(h []MHTHypothesis) []MHTHypothesis {
	sum := 0.0
	for i := range h {
		sum += h[i].Posterior
	}
	if sum <= 1e-12 {
		for i := range h {
			h[i].Posterior = 1.0 / float64(len(h))
		}
		return h
	}
	for i := range h {
		h[i].Posterior /= sum
	}
	return h
}

func pruneHyp(h []MHTHypothesis, n int) []MHTHypothesis {
	if len(h) <= n {
		return h
	}
	sort.Slice(h, func(i, j int) bool { return h[i].Posterior > h[j].Posterior })
	return h[:n]
}

func winnerHyp(h []MHTHypothesis) MHTHypothesis {
	if len(h) == 0 {
		return MHTHypothesis{ObjectID: 0, Variant: "uct", Posterior: 1}
	}
	w := h[0]
	for i := 1; i < len(h); i++ {
		if h[i].Posterior > w.Posterior {
			w = h[i]
		}
	}
	return w
}
