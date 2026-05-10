Perception — API reference
==========================

Shared perception primitives used by every tracker.

Public exports (``from perception import …``)
---------------------------------------------

.. autofunction:: perception.association.hungarian_associate

.. autofunction:: perception.association.oracle_associate

.. autoclass:: perception.association.AssociationResult
   :members:

.. autoclass:: perception.icp_pose.PoseEstimator
   :members:

.. autofunction:: perception.icp_pose.centroid_cam_from_mask

.. autofunction:: perception.visibility.visibility_p_v

Birth gating
------------

.. automodule:: perception.birth_gating
   :members:

Detection deduplication
-----------------------

.. automodule:: perception.det_dedup
   :members:

Voxel observability
-------------------

.. automodule:: perception.voxel_observability
   :members:

Adaptive kernels
----------------

.. automodule:: perception.adaptive_kernel
   :members:

Camera-pose refinement
----------------------

.. automodule:: perception.camera_pose_refiner
   :members:

Detection clients
-----------------

.. automodule:: perception.detection.det_client
   :members:

.. automodule:: perception.detection.mask_extractor
   :members:
