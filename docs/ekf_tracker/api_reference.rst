EKF tracker — API reference
===========================

This page is auto-generated from docstrings in the source.  For a
hand-written, narrative tour of the public surface, see
:doc:`api`.

Public facade
-------------

.. autoclass:: ekf_tracker.api.EkfTracker
   :members:
   :show-inheritance:

.. autoclass:: ekf_tracker.api.EkfObject
   :members:

.. autoclass:: ekf_tracker.api.SceneView
   :members:

Configuration dataclasses
-------------------------

.. autoclass:: ekf_tracker.config.BernoulliConfig
   :members:

.. autoclass:: ekf_tracker.config.TriggerConfig
   :members:

.. automodule:: ekf_tracker.configs
   :members: load_config, to_bernoulli_config, to_trigger_config

Key internals
-------------

The classes below are exposed publicly so advanced users can bypass the
``EkfTracker`` facade.  See :doc:`DISCUSSION` for the architectural rationale
and ``latex/bernoulli_ekf.pdf`` Part III for the algorithm-to-code map.

.. autoclass:: ekf_tracker.gaussian_ekf_tracker.GaussianEkfTracker
   :members:
   :show-inheritance:

.. autoclass:: ekf_tracker.orchestrator_gaussian.TwoTierOrchestratorGaussian
   :members:
   :show-inheritance:

.. autoclass:: ekf_tracker.perception_pipeline.LiveDetectionPipeline
   :members:

.. autoclass:: ekf_tracker.factor_graph.PoseGraphOptimizer
   :members:

.. automodule:: ekf_tracker.birth_gate
   :members:

State representation
~~~~~~~~~~~~~~~~~~~~

.. automodule:: ekf_tracker.state.gaussian_state
   :members:

.. automodule:: ekf_tracker.state.bernoulli
   :members:

.. automodule:: ekf_tracker.state.obs_chain
   :members:

Manipulation
~~~~~~~~~~~~

.. automodule:: ekf_tracker.manipulation.grasp_owner_detector
   :members:

.. automodule:: ekf_tracker.manipulation.gripper_state_inferrer
   :members:

.. automodule:: ekf_tracker.manipulation.gravity_predict
   :members:

Relations
~~~~~~~~~

.. automodule:: ekf_tracker.relations.relation_orchestrator
   :members:

.. automodule:: ekf_tracker.relations.relation_filter
   :members:

.. automodule:: ekf_tracker.relations.relation_client
   :members:

.. automodule:: ekf_tracker.relations.relation_utils
   :members:
