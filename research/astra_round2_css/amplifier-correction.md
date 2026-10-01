# Reopening even all-ones CSS amplifiers

This records a narrow correction to the family closure in
fieldnotes/2026-09-30-422-amplification-yield.md. It does not revise that
campaign's submitted matrices, retained witnesses, or verifier results.

For even a >= 4, take G_X=G_Z=(1,...,1), each a single row of length a.
The rows commute, both ranks are one, and k=a-2. Every weight-one operator
has nonzero syndrome; any weight-two vector is in the opposite kernel and
outside the own-check span {0,all-ones}. Hence this amplifier has distance 2,
including [[6,4,2]]. The original fieldnote's distance-1 statement does not
apply to these matrices.

The correct tensor product of two CSS chain complexes commutes directly from
the two factor commutation identities. Full row rank of the amplifier is
needed to eliminate the extra endpoint homology in the parameter formula; it
is not an extra requirement for commutation. Thus an implementation reporting
noncommutation for arbitrary valid CSS factors does not close that entire
construction family.

For the particular six-qubit amplifier used here, each logical representative
and its complement are disjoint and belong to the same class. Applying Lemma 4
of [arXiv:2609.37231](https://arxiv.org/abs/2609.37231) gives a factor-two
code-distance relation. This is a statement about the family, not a new
exact-distance certificate for any submitted base code.

The candidate built from codes/84-14-10.json passes the trusted gate as
[[574,56,20]], with maximum row weight 11. Its gate and explicit deep-search
witness are retained alongside research/astra_round2_css/reproduce.py.
Broader literature novelty remains unverified.

Research attribution: @mrvee-qC-bee; GPT-6 Astra, Codex, 2026-10-01.
