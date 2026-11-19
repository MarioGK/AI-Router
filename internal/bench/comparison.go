package bench

import (
	"fmt"
	"io"
	"maps"
	"slices"
	"text/tabwriter"

	"github.com/MarioGK/AI-Router/internal/decide"
)

// AgreementScore compares only trials where both deciders returned a usable
// answer to this question. Coverage keeps excluded errors and missing answers
// visible rather than treating them as agreement or silently reducing support.
type AgreementScore struct {
	Compared  int     `json:"compared"`
	Agreed    int     `json:"agreed"`
	Agreement float64 `json:"agreement"`
	Coverage  float64 `json:"coverage"`
}

// GroupComparison keeps the support for each question separate: a usable
// model choice does not imply a usable task-depth answer.
type GroupComparison struct {
	Name    string         `json:"name"`
	Planned int            `json:"planned"`
	Choice  AgreementScore `json:"choice"`
	Level   AgreementScore `json:"task_depth"`
	Joint   AgreementScore `json:"joint"`
	Model   AgreementScore `json:"final_model"`
}

// ComparisonDecision separates the classifier's own choice and task depth
// from the final model and effort selected by routing rules.
type ComparisonDecision struct {
	Choice        string             `json:"choice"`
	Level         string             `json:"task_depth"`
	Model         string             `json:"final_model"`
	Effort        string             `json:"applied_effort"`
	Probabilities map[string]float64 `json:"probabilities"`
	Levels        map[string]float64 `json:"depth_probabilities"`
	Policy        string             `json:"policy"`
}

// Discrepancy is evidence for a paired disagreement, without replacing the
// corpus's canonical labels with the reference's opinions. Probabilities come
// from the normalized primary attempt, never from a later pair question.
type Discrepancy struct {
	CaseName       string             `json:"case"`
	Trial          int                `json:"trial"`
	Group          string             `json:"group"`
	State          string             `json:"state"`
	Offered        []decide.Candidate `json:"offered"`
	Expect         []string           `json:"expect"`
	Reasoning      []string           `json:"reasoning"`
	ChoiceCompared bool               `json:"choice_compared"`
	LevelCompared  bool               `json:"depth_compared"`
	ModelCompared  bool               `json:"final_model_compared"`
	ChoiceDiffers  bool               `json:"choice_differs"`
	LevelDiffers   bool               `json:"depth_differs"`
	ModelDiffers   bool               `json:"final_model_differs"`
	Teacher        ComparisonDecision `json:"teacher"`
	Decider        ComparisonDecision `json:"decider"`
}

// Comparison measures agreement with a named reference, independently of
// canonical accuracy. Errors are excluded from paired denominators and
// recorded alongside coverage. Repeated trials retain their base identity.
type Comparison struct {
	Reference             string            `json:"reference"`
	Decider               string            `json:"decider"`
	Planned               int               `json:"planned"`
	UniqueCases           int               `json:"unique_cases"`
	ReferenceErrors       int               `json:"reference_errors"`
	DeciderErrors         int               `json:"decider_errors"`
	ReferenceCancelled    int               `json:"reference_cancelled"`
	DeciderCancelled      int               `json:"decider_cancelled"`
	ReferenceNotStarted   int               `json:"reference_not_started"`
	DeciderNotStarted     int               `json:"decider_not_started"`
	ReferenceMissingDepth int               `json:"reference_missing_depth"`
	DeciderMissingDepth   int               `json:"decider_missing_depth"`
	Choice                AgreementScore    `json:"choice"`
	Level                 AgreementScore    `json:"task_depth"`
	Joint                 AgreementScore    `json:"joint"`
	Model                 AgreementScore    `json:"final_model"`
	Groups                []GroupComparison `json:"groups"`
	Discrepancies         []Discrepancy     `json:"discrepancies"`
}

// Compare compares every other decider against reference. An empty reference
// disables comparison; an unknown or ambiguous name returns an error.
func (r Report) Compare(reference string) ([]Comparison, error) {
	if reference == "" {
		return nil, nil
	}
	referenceIndex := -1
	for j, name := range r.Deciders {
		if name != reference {
			continue
		}
		if referenceIndex >= 0 {
			return nil, fmt.Errorf("reference decider %q is ambiguous", reference)
		}
		referenceIndex = j
	}
	if referenceIndex < 0 {
		return nil, fmt.Errorf("reference decider %q is not in the report", reference)
	}
	comparisons := make([]Comparison, 0, len(r.Deciders)-1)
	for j, name := range r.Deciders {
		if j == referenceIndex {
			continue
		}
		comparison := Comparison{Reference: reference, Decider: name, Planned: len(r.Cases)}
		groups := map[string]*GroupComparison{}
		unique := map[string]bool{}
		for i, c := range r.Cases {
			if i >= len(r.Outcomes) || referenceIndex >= len(r.Outcomes[i]) || j >= len(r.Outcomes[i]) {
				return nil, fmt.Errorf("case %q trial %d has an incomplete outcome row", c.Name, c.Trial)
			}
			teacher, student := r.Outcomes[i][referenceIndex], r.Outcomes[i][j]
			comparison.addCoverage(teacher, student)
			unique[c.Name] = true
			groupName := caseGroup(c)
			group := groups[groupName]
			if group == nil {
				group = &GroupComparison{Name: groupName}
				groups[groupName] = group
			}
			group.Planned++
			choiceCompared := usableComparison(teacher) && usableComparison(student) && teacher.Choice != "" && student.Choice != ""
			levelCompared := usableComparison(teacher) && usableComparison(student) && teacher.Level != "" && student.Level != ""
			modelCompared := usableComparison(teacher) && usableComparison(student) && teacher.Model != "" && student.Model != ""
			choiceDiffers, levelDiffers, modelDiffers := teacher.Choice != student.Choice, teacher.Level != student.Level, teacher.Model != student.Model
			comparison.Choice.add(choiceCompared, !choiceDiffers)
			comparison.Level.add(levelCompared, !levelDiffers)
			comparison.Joint.add(choiceCompared && levelCompared, !choiceDiffers && !levelDiffers)
			comparison.Model.add(modelCompared, !modelDiffers)
			group.Choice.add(choiceCompared, !choiceDiffers)
			group.Level.add(levelCompared, !levelDiffers)
			group.Joint.add(choiceCompared && levelCompared, !choiceDiffers && !levelDiffers)
			group.Model.add(modelCompared, !modelDiffers)
			if !(choiceCompared && choiceDiffers || levelCompared && levelDiffers || modelCompared && modelDiffers) {
				continue
			}
			discrepancy := Discrepancy{CaseName: c.Name, Trial: c.Trial, Group: groupName,
				Expect: slices.Clone(c.Expect), Reasoning: slices.Clone(c.Reasoning),
				ChoiceCompared: choiceCompared, LevelCompared: levelCompared, ModelCompared: modelCompared,
				ChoiceDiffers: choiceCompared && choiceDiffers, LevelDiffers: levelCompared && levelDiffers, ModelDiffers: modelCompared && modelDiffers,
				Teacher: comparisonDecision(teacher), Decider: comparisonDecision(student)}
			if i < len(r.Inputs) {
				discrepancy.State = r.Inputs[i].State
				discrepancy.Offered = slices.Clone(r.Inputs[i].Candidates)
				for k := range discrepancy.Offered {
					discrepancy.Offered[k].Examples = slices.Clone(discrepancy.Offered[k].Examples)
				}
			}
			comparison.Discrepancies = append(comparison.Discrepancies, discrepancy)
		}
		comparison.UniqueCases = len(unique)
		comparison.Choice.finish(comparison.Planned)
		comparison.Level.finish(comparison.Planned)
		comparison.Joint.finish(comparison.Planned)
		comparison.Model.finish(comparison.Planned)
		for _, name := range slices.Sorted(maps.Keys(groups)) {
			group := *groups[name]
			group.Choice.finish(group.Planned)
			group.Level.finish(group.Planned)
			group.Joint.finish(group.Planned)
			group.Model.finish(group.Planned)
			comparison.Groups = append(comparison.Groups, group)
		}
		comparisons = append(comparisons, comparison)
	}
	return comparisons, nil
}

func usableComparison(o Outcome) bool { return o.Started && !o.Cancelled && o.Err == "" }

func (s *AgreementScore) add(compared, agreed bool) {
	if compared {
		s.Compared++
		if agreed {
			s.Agreed++
		}
	}
}

func (s *AgreementScore) finish(planned int) {
	s.Agreement, s.Coverage = share(s.Agreed, s.Compared), share(s.Compared, planned)
}

func (c *Comparison) addCoverage(teacher, student Outcome) {
	if teacher.Err != "" {
		c.ReferenceErrors++
	}
	if student.Err != "" {
		c.DeciderErrors++
	}
	if teacher.Cancelled {
		c.ReferenceCancelled++
	}
	if student.Cancelled {
		c.DeciderCancelled++
	}
	if !teacher.Started {
		c.ReferenceNotStarted++
	}
	if !student.Started {
		c.DeciderNotStarted++
	}
	if usableComparison(teacher) && teacher.Level == "" {
		c.ReferenceMissingDepth++
	}
	if usableComparison(student) && student.Level == "" {
		c.DeciderMissingDepth++
	}
}

func comparisonDecision(o Outcome) ComparisonDecision {
	d := ComparisonDecision{Choice: o.Choice, Level: o.Level, Model: o.Model, Effort: o.Effort, Policy: o.Policy}
	for _, attempt := range o.Attempts {
		if len(attempt.Pair) == 0 && (attempt.Outcome == decide.Decided || attempt.Outcome == decide.Doubted) {
			d.Probabilities, d.Levels = maps.Clone(attempt.Probabilities), maps.Clone(attempt.Levels)
			break
		}
	}
	return d
}

func (r Report) writeComparisons(w io.Writer) {
	if r.Reference == "" {
		return
	}
	comparisons, err := r.Compare(r.Reference)
	if err != nil {
		fmt.Fprintf(w, "\nreference comparison error: %v\n", err)
		return
	}
	fmt.Fprintf(w, "\nreference comparison: %s (agreement with the reference is separate from canonical accuracy)\n", r.Reference)
	fmt.Fprintln(w, "Choice compares the primary verdict before economy or escalation; depth compares task thinking, final model compares routed targets.")
	table := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(table, "decider\tgroup\tchoice agreed/paired\tdepth agreed/paired\tjoint agreed/paired\tfinal model agreed/paired\tpaired choice coverage")
	for _, comparison := range comparisons {
		writeComparisonRow(table, comparison.Decider, "all", comparison.Choice, comparison.Level, comparison.Joint, comparison.Model)
		for _, group := range comparison.Groups {
			writeComparisonRow(table, comparison.Decider, group.Name, group.Choice, group.Level, group.Joint, group.Model)
		}
	}
	table.Flush()
	for _, comparison := range comparisons {
		fmt.Fprintf(w, "%s: %d planned trials / %d unique cases; errors reference=%d decider=%d; missing depth reference=%d decider=%d\n",
			comparison.Decider, comparison.Planned, comparison.UniqueCases, comparison.ReferenceErrors, comparison.DeciderErrors,
			comparison.ReferenceMissingDepth, comparison.DeciderMissingDepth)
		for _, d := range comparison.Discrepancies {
			fmt.Fprintf(w, "disagreement %s/%s trial %d: reference choice=%s depth=%s final=%s effort=%s; decider choice=%s depth=%s final=%s effort=%s\n",
				comparison.Decider, d.CaseName, d.Trial, d.Teacher.Choice, d.Teacher.Level, d.Teacher.Model, d.Teacher.Effort,
				d.Decider.Choice, d.Decider.Level, d.Decider.Model, d.Decider.Effort)
		}
	}
}

func writeComparisonRow(w io.Writer, decider, group string, choice, level, joint, model AgreementScore) {
	fmt.Fprintf(w, "%s\t%s\t%d/%d (%.0f%%)\t%d/%d (%.0f%%)\t%d/%d (%.0f%%)\t%d/%d (%.0f%%)\t%.0f%%\n",
		decider, group, choice.Agreed, choice.Compared, 100*choice.Agreement, level.Agreed, level.Compared, 100*level.Agreement,
		joint.Agreed, joint.Compared, 100*joint.Agreement, model.Agreed, model.Compared, 100*model.Agreement, 100*choice.Coverage)
}
