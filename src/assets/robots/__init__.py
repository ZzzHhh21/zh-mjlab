from importlib.util import find_spec

if find_spec(__name__ + ".unitree_g1_wuji.g1_constants") is not None:
  from .unitree_g1_wuji.g1_constants import (
    G1_ACTION_SCALE as G1_ACTION_SCALE,
  )
  from .unitree_g1_wuji.g1_constants import (
    get_g1_robot_cfg as get_g1_robot_cfg,
  )

if find_spec(__name__ + ".unitree_g1_wuji.g1_23dof_constants") is not None:
  from .unitree_g1_wuji.g1_23dof_constants import (
    G1_23DOF_ACTION_SCALE as G1_23DOF_ACTION_SCALE,
  )
  from .unitree_g1_wuji.g1_23dof_constants import (
    get_g1_23dof_robot_cfg as get_g1_23dof_robot_cfg,
  )

from .unitree_g1_wuji.g1_wuji_manip_constants import (
  G1_WUJI_MANIP_ACTION_SCALE as G1_WUJI_MANIP_ACTION_SCALE,
)
from .unitree_g1_wuji.g1_wuji_manip_constants import (
  get_g1_wuji_manip_robot_cfg as get_g1_wuji_manip_robot_cfg,
)

from .unitree_r1.r1_constants import (
  R1_ACTION_SCALE as R1_ACTION_SCALE,
)
from .unitree_r1.r1_constants import (
  get_r1_robot_cfg as get_r1_robot_cfg,
)

from .unitree_h1_2.h1_2_constants import (
  H1_2_ACTION_SCALE as H1_2_ACTION_SCALE,
)
from .unitree_h1_2.h1_2_constants import (
  get_h1_2_robot_cfg as get_h1_2_robot_cfg,
)

from .unitree_h2.h2_constants import (
  H2_ACTION_SCALE as H2_ACTION_SCALE,
)
from .unitree_h2.h2_constants import (
  get_h2_robot_cfg as get_h2_robot_cfg,
)

from .i2rt_yam.yam_constants import YAM_ACTION_SCALE as YAM_ACTION_SCALE
from .i2rt_yam.yam_constants import get_yam_robot_cfg as get_yam_robot_cfg
