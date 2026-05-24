"""
Centralized imports for the TODDLERS project.

This module organizes all imports used throughout the project following PEP 8 guidelines.
Imports are grouped by category and ordered alphabetically within each group.
"""

# Standard Library Imports
## System and OS
import os
import sys
import subprocess
import shutil
import platform

## Data Structures and Types
from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Tuple,
    Any,
    Optional,
    NamedTuple,
    Set
)
from enum import Enum
from abc import ABC, abstractmethod
from functools import wraps

## File Operations and I/O
import json
import pickle
import io
import tempfile

## Text Processing
import re

## Date and Time
import time
from datetime import datetime
import uuid

## Logging and Debugging
import logging
import signal
import warnings
import traceback

## Concurrency and Threading
import queue
from queue import Queue
from threading import Thread, current_thread, RLock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import (
    Process,
    Pool,
    cpu_count,
    Lock,
    Manager,
    Value,
    current_process,
    get_context,
    Event
)

## Context Management
from contextlib import contextmanager

# Third-Party Imports
## Scientific Computing
import numpy as np
import scipy
from scipy import constants
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import (
    solve_ivp,
    quad,
    odeint,
    simpson,
    cumulative_trapezoid,
    IntegrationWarning
)
from scipy.interpolate import (
    interp1d,
    RegularGridInterpolator,
    InterpolatedUnivariateSpline as InterpUS
)
from scipy.signal import find_peaks
from scipy.optimize import (
    brentq,
    fsolve,
    root_scalar,
    minimize_scalar,
    newton
)

## Astronomy
import astropy.constants as const
import astropy.units as u

## Visualization
import matplotlib.pyplot as plt

## System Monitoring
import psutil

## File System
from pathlib import Path

# Version Information
# print(f'scipy: {scipy.version}')

# Local Imports
from .constants import *

# Type Aliases
Array = np.ndarray