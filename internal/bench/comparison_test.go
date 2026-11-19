package bench

import (
	"bytes"
	"encoding/json/v2"
	"strings"
	"testing"

	"github.com/MarioGK/AI-Router/internal/decide"
)

func TestCompareSeparatesPrimaryVerdictFromEconomyAndPair(t *testing.T) {
	teacher := Outcome{Started: true, Choice: "cheap", Level: "high", Model: "hard", Effort: "xhigh", Policy: "teacher-policy",
		Attempts: []decide.Attempt{
			{Outcome: decide.Failed, Probabilities: map[string]float64{"hard": 1}},
			{Outcome: decide.Doubted, Choice: "cheap", Probabilities: map[string]float64{"cheap": 0.6, "hard": 0.4}, Levels: map[string]float64{"high": 1}},
			{Outcome: decide.Decided, Pair: []string{"cheap", "hard"}, Probabilities: map[string]float64{"cheap": 0.1, "hard": 0.9}},
		}}
	student := Outcome{Started: true, Choice: "hard", Level: "high", Model: "hard", Effort: "high"}
	r := Report{Reference: "sol61", Deciders: []string{"jev", "sol61"},
		Cases:    []Case{{Name: "complex/security", Trial: 2, Expect: []string{"hard"}, Reasoning: []string{"high"}, Prompt: "original request"}},
		Inputs:   []Input{{State: "prepared request", Candidates: []decide.Candidate{{ID: "cheap", Examples: []string{"mechanical"}}, {ID: "hard"}}}},
		Outcomes: [][]Outcome{{student, teacher}}}
	comparisons, err := r.Compare("sol61")
	if err != nil || len(comparisons) != 1 {
		t.Fatalf("Compare = %+v, %v", comparisons, err)
	}
	c := comparisons[0]
	if c.Choice.Compared != 1 || c.Choice.Agreed != 0 || c.Level.Agreed != 1 || c.Joint.Agreed != 0 || c.Model.Agreed != 1 {
		t.Fatalf("primary and final agreement mixed = %+v", c)
	}
	if !r.Correct(0, 0) || !r.Correct(0, 1) || len(c.Discrepancies) != 1 {
		t.Fatal("comparison changed canonical accuracy or lost discrepancy")
	}
	d := c.Discrepancies[0]
	if d.CaseName != "complex/security" || d.Trial != 2 || d.State != "prepared request" ||
		!d.ChoiceDiffers || d.LevelDiffers || d.ModelDiffers || d.Teacher.Probabilities["cheap"] != 0.6 ||
		d.Teacher.Levels["high"] != 1 || d.Teacher.Effort != "xhigh" || d.Decider.Effort != "high" || len(d.Offered) != 2 {
		t.Fatalf("discrepancy evidence = %+v", d)
	}
	d.Teacher.Probabilities["cheap"] = 0
	d.Offered[0].Examples[0] = "changed"
	d.Expect[0] = "changed"
	if r.Outcomes[0][1].Attempts[1].Probabilities["cheap"] != 0.6 || r.Inputs[0].Candidates[0].Examples[0] != "mechanical" || r.Cases[0].Expect[0] != "hard" {
		t.Fatal("comparison evidence aliases the report")
	}
}

func TestCompareExcludesErrorsAndMissingAnswersWithCoverage(t *testing.T) {
	valid := Outcome{Started: true, Choice: "cheap", Level: "high", Model: "cheap"}
	noDepth := valid
	noDepth.Level = ""
	failed := Outcome{Started: true, Err: "unavailable"}
	cancelled := Outcome{Cancelled: true, Err: "context canceled"}
	cancelledStarted := Outcome{Started: true, Cancelled: true, Err: "context canceled"}
	r := Report{Deciders: []string{"sol61", "jev"},
		Cases: []Case{
			{Name: "paired", Group: "valid"}, {Name: "student-fails", Group: "excluded"},
			{Name: "no-depth", Group: "valid"}, {Name: "teacher-fails", Group: "excluded"},
			{Name: "teacher-cancelled", Group: "excluded"}, {Name: "student-cancelled", Group: "excluded"},
		},
		Outcomes: [][]Outcome{{valid, valid}, {valid, failed}, {noDepth, valid}, {failed, valid}, {cancelled, valid}, {valid, cancelledStarted}}}
	comparisons, err := r.Compare("sol61")
	if err != nil {
		t.Fatal(err)
	}
	c := comparisons[0]
	if c.Planned != 6 || c.Choice.Compared != 2 || c.Choice.Agreed != 2 || c.Choice.Coverage != 1.0/3 ||
		c.Level.Compared != 1 || c.Level.Coverage != 1.0/6 || c.Joint.Compared != 1 || c.Model.Compared != 2 {
		t.Fatalf("paired denominators = %+v", c)
	}
	if c.ReferenceErrors != 2 || c.DeciderErrors != 2 || c.ReferenceCancelled != 1 || c.DeciderCancelled != 1 ||
		c.ReferenceNotStarted != 1 || c.DeciderNotStarted != 0 || c.ReferenceMissingDepth != 1 || c.DeciderMissingDepth != 0 {
		t.Fatalf("excluded trial coverage = %+v", c)
	}
	if len(c.Discrepancies) != 0 || len(c.Groups) != 2 || c.Groups[0].Name != "excluded" || c.Groups[0].Planned != 4 ||
		c.Groups[0].Choice.Compared != 0 || c.Groups[0].Choice.Agreement != 0 || c.Groups[1].Choice.Coverage != 1 || c.Groups[1].Level.Coverage != 0.5 {
		t.Fatalf("group support = %+v", c.Groups)
	}
}

func TestCompareRepeatEvidenceRetainsTaskIdentity(t *testing.T) {
	r := Report{Reference: "sol61", Deciders: []string{"sol61", "jev", "luna"}}
	for trial := range 2 {
		r.Cases = append(r.Cases, Case{Name: "same/task", Trial: trial + 1})
		r.Outcomes = append(r.Outcomes, []Outcome{
			{Started: true, Choice: "hard", Level: "high", Model: "hard"},
			{Started: true, Choice: "cheap", Level: "low", Model: "cheap"},
			{Started: true, Choice: "hard", Level: "high", Model: "hard"},
		})
	}
	comparisons, err := r.Compare("sol61")
	if err != nil || len(comparisons) != 2 {
		t.Fatalf("Compare = %+v, %v", comparisons, err)
	}
	jev, luna := comparisons[0], comparisons[1]
	if jev.Planned != 2 || jev.UniqueCases != 1 || jev.Joint.Compared != 2 || len(jev.Discrepancies) != 2 ||
		jev.Discrepancies[0].Trial != 1 || jev.Discrepancies[1].Trial != 2 || luna.Joint.Agreement != 1 || len(luna.Discrepancies) != 0 {
		t.Fatalf("repeat identity = %+v", comparisons)
	}
	var out bytes.Buffer
	if err := r.WriteJSON(&out); err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Reference   string       `json:"reference"`
		Comparisons []Comparison `json:"comparisons"`
	}
	if err := json.Unmarshal(out.Bytes(), &decoded); err != nil || decoded.Reference != "sol61" || len(decoded.Comparisons) != 2 || decoded.Comparisons[0].UniqueCases != 1 {
		t.Fatalf("JSON comparison = %s, %v", out.String(), err)
	}
	out.Reset()
	r.writeComparisons(&out)
	if !strings.Contains(out.String(), "reference comparison: sol61") || !strings.Contains(out.String(), "disagreement jev/same/task trial 2") ||
		!strings.Contains(out.String(), "before economy or escalation") {
		t.Fatalf("comparison text = %s", out.String())
	}
}

func TestCompareValidatesReferenceAndOutcomeRows(t *testing.T) {
	r := Report{Deciders: []string{"sol61", "jev"}}
	if comparisons, err := r.Compare(""); err != nil || len(comparisons) != 0 {
		t.Fatalf("disabled comparison = %+v, %v", comparisons, err)
	}
	if _, err := r.Compare("unknown"); err == nil {
		t.Fatal("unknown teacher accepted")
	}
	r.Deciders = []string{"sol61", "sol61"}
	if _, err := r.Compare("sol61"); err == nil {
		t.Fatal("ambiguous teacher accepted")
	}
	r.Deciders = []string{"sol61", "jev"}
	r.Cases = []Case{{Name: "incomplete"}}
	if _, err := r.Compare("sol61"); err == nil {
		t.Fatal("missing comparison row accepted")
	}
	r.Reference = "unknown"
	var out bytes.Buffer
	if err := r.WriteJSON(&out); err == nil || out.Len() != 0 {
		t.Fatal("JSON silently omitted an invalid reference")
	}
}
