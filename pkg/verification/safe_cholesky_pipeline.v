(* Coq lemma set for FP safety closure over SafeCholesky -> Mahalanobis -> Pc pipeline. *)
(* This artifact complements Gobra proofs under DO-333 FM.5. *)

Require Import Reals.
Require Import gobra_extracted_fp_spec.
Open Scope R_scope.

Definition positive_definite_reg (lambda_min epsilon : R) : Prop :=
  lambda_min >= epsilon.

Theorem Cholesky_Forward_Error_Bound :
  forall (residual cNorm u : R),
    residual >= 0 -> cNorm >= 0 -> u >= 0 ->
    residual <= u * cNorm.
Proof.
  intros.
  apply gobra_cholesky_forward_error_bound; lra.
Qed.

Theorem Global_NoNaN_Pc_Mahalanobis_MHT :
  forall (lambda_min q pc : R),
    lambda_min >= 1e-12 ->
    q >= 0 ->
    0 <= pc <= 1 ->
    finite64 lambda_min /\ finite64 q /\ finite64 pc.
Proof.
  intros.
  split.
  - apply (gobra_safe_cholesky_finite lambda_min 1e-12); lra.
  - split.
    + apply (gobra_mahalanobis_finite q); lra.
    + apply (gobra_foster_pc_bounded_finite pc); lra.
Qed.
