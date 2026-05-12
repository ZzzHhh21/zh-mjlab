from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = [
  "utils",
  ".mdp",
  "src.tasks.tracking.config.g1",
  "src.tasks.velocity.config.g1",
]

import_packages(__name__, _BLACKLIST_PKGS)
