import os, sys

thisdir = os.path.dirname(__file__)
libdir_server = os.path.join(thisdir, '..')
libdir_gpa = os.path.join(thisdir, '../src/gpa_package_server/lib')
bindir_gpa = os.path.join(thisdir, '../src/gpa_package_server/bin')
libdir_vpa = os.path.join(thisdir, '../src/view-point-server')

if libdir_server not in sys.path:
    sys.path.insert(0, libdir_server)

if libdir_gpa not in sys.path:
    sys.path.insert(0, libdir_gpa)

if bindir_gpa not in sys.path:
    sys.path.insert(0, bindir_gpa)

if libdir_vpa not in sys.path:
    sys.path.insert(0, libdir_vpa)

