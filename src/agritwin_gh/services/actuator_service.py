"""
ActuatorService — manages actuator state and manual override application.

Responsibility
--------------
* Read current actuator levels from ``RuntimeStore`` and format them as
  ``ActuatorStateResponse``.
* Validate and apply batch actuator overrides from ``POST /api/actuators/set``.
* After each MPC solve, receive the new actuator trajectory and update the
  store so the API reflects the latest recommendations.

Modules reused from the MPC layer
----------------------------------
* ``agritwin_gh.mpc.state.ActuatorState``          — core actuator dataclass.
* ``agritwin_gh.mpc.constants.CONTROL_VARIABLES``  — canonical actuator order.
* ``agritwin_gh.mpc.constraints``                  — actuator bounds for validation.
"""

from __future__ import annotations

import logging

from agritwin_gh.core.runtime_store import RuntimeStore

logger = logging.getLogger("agritwin.services.actuator")


class ActuatorService:
    """Manages actuator state and manual overrides."""

    def __init__(self, store: RuntimeStore) -> None:
        self._store = store

    def get_state(self):
        """Return current actuator states as ``ActuatorStateResponse``."""
        return self._store.as_actuator_state_response()

    def apply_overrides(self, overrides: list) -> list[str]:
        """Validate and apply a list of actuator level overrides.

        Delegates to ``ControlService.apply_actuator_set()`` which validates
        actuator IDs, clamps levels to [0, 100], and writes an ``OverrideConfig``
        to the ``RuntimeStore``.

        Parameters
        ----------
        overrides:
            List of ``ActuatorSetEntry`` objects from the request body.

        Returns
        -------
        list[str]
            IDs of actuators that were successfully updated.
        """
        from agritwin_gh.schemas.actuator_schemas import ActuatorSetRequest
        from agritwin_gh.services.control_service import ControlService
        request = ActuatorSetRequest(actuators=overrides)
        result = ControlService(store=self._store).apply_actuator_set(request)
        return result.overridden

    def update_from_mpc_solution(self, solution: object) -> None:
        """No-op stub — MPC solution application is now handled by ``LoopService``.

        ``LoopService.run_one_step()`` reads the ``DTLoopStepResult.action_applied``
        field directly and updates the store via ``update_from_step_result()``.
        This method is retained for backward-compatibility with any callers that
        reference ``ActuatorService`` but should not be used in new code.
        """
        logger.debug("ActuatorService.update_from_mpc_solution: delegated to LoopService.")
