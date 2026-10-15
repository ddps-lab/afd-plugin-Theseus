# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU-specific AFD connector implementations."""

from afd_plugin.connectors.gpu.p2p import P2pNcclAFDConnector
from afd_plugin.connectors.gpu.routed import P2pNcclRoutedAFDConnector

__all__ = ["P2pNcclAFDConnector", "P2pNcclRoutedAFDConnector"]
