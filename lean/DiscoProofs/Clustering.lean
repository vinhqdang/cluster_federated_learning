/-
Theorem 2 (safety, maximality, exact recovery): the combinatorial argument.

The clustering is modelled abstractly. Clients have type `α`; `conf i j` says that
clients `i` and `j` conflict (hold a common class with different class-conditional
distributions). A state of the algorithm is a list of clusters; a merge step replaces
two clusters `A, B` of the state by `A ∪ B`, and is only allowed when `linked A B`
(the linkage reaches the threshold). The analytic Claims 1 and 2 (file `Linkage.lean`)
are the hypotheses `hclaim1`, `hclaim2` below. The results hold for every merge order.
-/
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Logic.Relation

namespace DiscoProofs

variable {α : Type*} [DecidableEq α]

/-- A cluster is conflict-free if no two of its members conflict. -/
def ConflictFree (conf : α → α → Prop) (A : Finset α) : Prop :=
  ∀ i ∈ A, ∀ j ∈ A, ¬ conf i j

/-- One merge step of the agglomerative procedure. -/
inductive MergeStep (linked : Finset α → Finset α → Prop) :
    List (Finset α) → List (Finset α) → Prop
  | merge (P : List (Finset α)) (A B : Finset α) (hA : A ∈ P) (hB : B ∈ P)
      (hl : linked A B) : MergeStep linked P ((A ∪ B) :: ((P.erase A).erase B))

omit [DecidableEq α] in
theorem singleton_conflict_free (conf : α → α → Prop) (hirr : ∀ i, ¬ conf i i) (i : α) :
    ConflictFree conf {i} := by
  intro x hx y hy
  rw [Finset.mem_singleton] at hx hy
  rw [hx, hy]
  exact hirr i

/-- A merge allowed by the linkage never creates a conflicting cluster (Claim 1). -/
theorem merge_preserves_conflict_free (conf : α → α → Prop)
    (linked : Finset α → Finset α → Prop)
    (hclaim1 : ∀ A B, ConflictFree conf A → ConflictFree conf B →
      ¬ ConflictFree conf (A ∪ B) → ¬ linked A B)
    (A B : Finset α) (hA : ConflictFree conf A) (hB : ConflictFree conf B)
    (hl : linked A B) : ConflictFree conf (A ∪ B) := by
  by_contra h
  exact hclaim1 A B hA hB h hl

/-- Theorem 2(a), safety: starting from conflict-free clusters (e.g. singletons), every
state reachable by any sequence of allowed merges consists of conflict-free clusters. -/
theorem safety_all_merges (conf : α → α → Prop) (linked : Finset α → Finset α → Prop)
    (hclaim1 : ∀ A B, ConflictFree conf A → ConflictFree conf B →
      ¬ ConflictFree conf (A ∪ B) → ¬ linked A B)
    (P₀ P : List (Finset α)) (h0 : ∀ C ∈ P₀, ConflictFree conf C)
    (hreach : Relation.ReflTransGen (MergeStep linked) P₀ P) :
    ∀ C ∈ P, ConflictFree conf C := by
  induction hreach with
  | refl => exact h0
  | tail _ hstep ih =>
    cases hstep with
    | merge A B hA hB hl =>
      intro C hC
      rcases List.mem_cons.mp hC with h | h
      · subst h
        exact merge_preserves_conflict_free conf linked hclaim1 A B (ih A hA) (ih B hB) hl
      · exact ih C (List.mem_of_mem_erase (List.mem_of_mem_erase h))

/-- Theorem 2(b), maximality: if the procedure stopped (no two distinct clusters are
linked) and compatible clusters with enough shared evidence are always linked (Claim 2),
then any two distinct output clusters with enough evidence contain a conflicting pair. -/
theorem maximality (conf : α → α → Prop) (linked evid : Finset α → Finset α → Prop)
    (hclaim2 : ∀ A B, ConflictFree conf (A ∪ B) → evid A B → linked A B)
    (P : List (Finset α)) (hterm : ∀ A ∈ P, ∀ B ∈ P, A ≠ B → ¬ linked A B) :
    ∀ A ∈ P, ∀ B ∈ P, A ≠ B → evid A B → ¬ ConflictFree conf (A ∪ B) := by
  intro A hA B hB hne he hcf
  exact hterm A hA B hB hne (hclaim2 A B hcf he)

/-- Theorem 2(c), exact recovery: under full class support, two clients conflict exactly
when they belong to different concept groups. Then every partition of the clients into
conflict-free clusters that is maximal (no two clusters can be merged without a conflict)
coincides with the partition into groups: two clients share an output cluster if and
only if they belong to the same group. -/
theorem exact_recovery {G : Type*} (group : α → G) (conf : α → α → Prop)
    (hconf : ∀ i j, conf i j ↔ group i ≠ group j)
    (P : List (Finset α)) (hcf : ∀ C ∈ P, ConflictFree conf C)
    (hmax : ∀ A ∈ P, ∀ B ∈ P, A ≠ B → ¬ ConflictFree conf (A ∪ B))
    (i j : α) (hi : ∃ A ∈ P, i ∈ A) (hj : ∃ B ∈ P, j ∈ B) :
    (∃ C ∈ P, i ∈ C ∧ j ∈ C) ↔ group i = group j := by
  -- purity: every member of a conflict-free cluster has the same group
  have pure : ∀ C ∈ P, ∀ x ∈ C, ∀ y ∈ C, group x = group y := by
    intro C hC x hx y hy
    by_contra hne
    exact hcf C hC x hx y hy ((hconf x y).mpr hne)
  constructor
  · rintro ⟨C, hC, hiC, hjC⟩
    exact pure C hC i hiC j hjC
  · intro hg
    obtain ⟨A, hA, hiA⟩ := hi
    obtain ⟨B, hB, hjB⟩ := hj
    by_cases hAB : A = B
    · subst hAB
      exact ⟨A, hA, hiA, hjB⟩
    · exfalso
      apply hmax A hA B hB hAB
      intro x hx y hy hxy
      have gx : group x = group i := by
        rcases Finset.mem_union.mp hx with h | h
        · exact pure A hA x h i hiA
        · exact (pure B hB x h j hjB).trans hg.symm
      have gy : group y = group i := by
        rcases Finset.mem_union.mp hy with h | h
        · exact pure A hA y h i hiA
        · exact (pure B hB y h j hjB).trans hg.symm
      exact (hconf x y).mp hxy (gx.trans gy.symm)

end DiscoProofs
