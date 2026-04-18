(* Gobra-extracted floating-point safety interface for Coq composition proofs. *)
(* Generated from the contract surface in pkg/verification/gobra_specs.gobra. *)

Require Import Reals.
Open Scope R_scope.

Parameter finite64 : R -> Prop.

Parameter gobra_safe_cholesky_finite : forall (lambda_min epsilon : R),
  epsilon >= 1e-12 ->
  lambda_min >= epsilon ->
  finite64 lambda_min.

Parameter gobra_mahalanobis_finite : forall (q : R),
  q >= 0 ->
  finite64 q.

Parameter gobra_foster_pc_bounded_finite : forall (pc : R),
  0 <= pc <= 1 ->
  finite64 pc.

Parameter gobra_cholesky_forward_error_bound : forall (residual cNorm u : R),
  residual >= 0 -> cNorm >= 0 -> u >= 0 ->
  residual <= u * cNorm.
