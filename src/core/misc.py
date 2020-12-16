import logging
import os
import os.path as osp
import sys
from time import localtime
from collections import OrderedDict, deque
from weakref import proxy


FORMAT_LONG = "[%(asctime)-15s %(funcName)s] %(message)s"
FORMAT_SHORT = "%(message)s"


class _LessThanFilter(logging.Filter):
    def __init__(self, max_level, name=''):
        super().__init__(name=name)
        self.max_level = getattr(logging, max_level.upper()) if isinstance(max_level, str) else int(max_level)
    def filter(self, record):
        return record.levelno < self.max_level


class Logger:
    _count = 0

    def __init__(self, scrn=True, log_dir='', phase=''):
        super().__init__()
        self._logger = logging.getLogger('logger_{}'.format(Logger._count))
        Logger._count += 1
        self._logger.setLevel(logging.DEBUG)

        self._err_handler = logging.StreamHandler(stream=sys.stderr)
        self._err_handler.setLevel(logging.WARNING)
        self._err_handler.setFormatter(logging.Formatter(fmt=FORMAT_SHORT))
        self._logger.addHandler(self._err_handler)

        if scrn:
            self._scrn_handler = logging.StreamHandler(stream=sys.stdout)
            self._scrn_handler.setLevel(logging.INFO)
            self._scrn_handler.addFilter(_LessThanFilter(logging.WARNING))
            self._scrn_handler.setFormatter(logging.Formatter(fmt=FORMAT_SHORT))
            self._logger.addHandler(self._scrn_handler)
            
        if log_dir and phase:
            self.log_path = osp.join(log_dir,
                    "{}-{:-4d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}.log".format(
                        phase, *localtime()[:6]
                      ))
            self.show_nl("Log into {}\n\n".format(self.log_path))
            self._file_handler = logging.FileHandler(filename=self.log_path)
            self._file_handler.setLevel(logging.DEBUG)
            self._file_handler.setFormatter(logging.Formatter(fmt=FORMAT_LONG))
            self._logger.addHandler(self._file_handler)

    def show(self, *args, **kwargs):
        return self._logger.info(*args, **kwargs)

    def show_nl(self, *args, **kwargs):
        self._logger.info("")
        return self.show(*args, **kwargs)

    def dump(self, *args, **kwargs):
        return self._logger.debug(*args, **kwargs)

    def warn(self, *args, **kwargs):
        return self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self._logger.error(*args, **kwargs)

    def fatal(self, *args, **kwargs):
        return self._logger.critical(*args, **kwargs)

_logger = Logger()


class _WeakAttribute:
    def __get__(self, instance, owner):
        return instance.__dict__[self.name]
    def __set__(self, instance, value):
        if value is not None:
            value = proxy(value)
        instance.__dict__[self.name] = value
    def __set_name__(self, owner, name):
        self.name = name


class _TreeNode:
    parent = _WeakAttribute()   # To avoid circular reference

    def __init__(
        self, name, value=None, parent=None, children=None,
        sep='/', none_val=None
    ):
        super().__init__()
        self.name = name
        self.val = value
        self.parent = parent
        self.children = children if isinstance(children, dict) else {}
        if isinstance(children, list):
            for child in children:
                self._add_child(child)
        self.path = name
        self._sep = sep
        self._none = none_val
    
    def get_child(self, name):
        return self.children.get(name, None)

    def add_placeholder(self, name):
        return self.add_child(name, value=self._none)

    def add_child(self, name, value, warning=False):
        r"""If node does not exist or is a placeholder, create it;
        otherwise skip and return the existing node.
        """
        child = self.get_child(name)
        if child is None:
            child = _TreeNode(name, value, parent=self, sep=self._sep, none_val=self._none)
            self._add_child(child)
        elif child.is_placeholder():
            # Retain the links of a placeholder,
            # i.e. just fill in it.
            child.val = value
        else:
            if warning: 
                _logger.warn("Node already exists.")
        return child

    def is_leaf(self):
        return len(self.children) == 0

    def is_placeholder(self):
        return self.val == self._none

    def __repr__(self):
        try:
            repr = self.path + " " + str(self.val)
        except TypeError:
            repr = self.path
        return repr

    def __contains__(self, name):
        return name in self.children.keys()

    def __getitem__(self, key):
        return self.get_child(key)

    def _add_child(self, node):
        r"""Add a child node into self.children.
        If the node already exists, just update its information.
        """
        self.children.update({
            node.name: node
        })
        node.path = self._sep.join([self.path, node.name])
        node.parent = self

    def apply(self, func):
        r"""Apply a callback function to ALL descendants.
        This is useful for recursive traversal.
        """
        ret = [func(self)]
        for _, node in self.children.items():
            ret.extend(node.apply(func))
        return ret

    def bfs_tracker(self):
        queue = deque()
        queue.append(self)
        while(queue):
            curr = queue.popleft()
            yield curr
            if curr.is_leaf():
                continue
            for c in curr.children.values():
                queue.append(c)


class _Tree:
    def __init__(
        self, name, value=None, eles=None, 
        sep='/', none_val=None
    ):
        super().__init__()
        self._sep = sep
        self._none = none_val
        
        self.root = _TreeNode(name, value, parent=None, children={}, sep=self._sep, none_val=self._none)
        if eles is not None:
            assert isinstance(eles, dict)
            self.build_tree(OrderedDict(eles or {}))

    def build_tree(self, elements):
        # The order of the siblings is not retained
        for path, ele in elements.items():
            self.add_node(path, ele)

    def __repr__(self):
        _str = ""
        # DFS
        stack = []
        stack.append((self.root, 0))
        while(stack):
            root, layer = stack.pop()
            _str += " "*layer + "-" + root.__repr__() + "\n"

            if root.is_leaf():
                continue
            # Note that the siblings are printed in alphabetical order.
            for c in sorted(list(root.children.values()), key=lambda n: n.name, reverse=True):
                stack.append((c, layer+1))

        return _str

    def __contains__(self, obj):
        return any(self.perform(lambda node: obj in node))

    def perform(self, func):
        return self.root.apply(func)

    def get_node(self, tar, mode='name'):
        r"""This is different from a travasal in that this search allows early stop."""
        if mode not in ('name', 'path', 'val'):
            raise NotImplementedError("Invalid mode")
        if mode == 'path':
            nodes = self.parse_path(tar)
            root = self.root
            for r in nodes:
                if root is None:
                    break
                root = root.get_child(r)
            return root
        else:
            # BFS
            bfs_tracker = self.root.bfs_tracker()
            # bfs_tracker.send(None)

            for node in bfs_tracker:
                if getattr(node, mode) == tar:
                    return node
            return None

    def add_node(self, path, val):
        if not path.strip():
            raise ValueError("The path is null.")
        path = path.rstrip(self._sep)
        names = self.parse_path(path)
        root = self.root
        nodes = [root]
        for name in names[:-1]:
            # Add a placeholder or skip an existing node
            root = root.add_placeholder(name)
            nodes.append(root)
        root = root.add_child(names[-1], val, True)
        return root, nodes

    def parse_path(self, path):
        return path.split(self._sep)

    def join(self, *args):
        return self._sep.join(args)
        
        
class OutPathGetter:
    def __init__(self, root='', log='logs', out='out', weight='weights', suffix='', **subs):
        super().__init__()
        self._root = root.rstrip(os.sep)    # Work robustly on multiple ending '/'s
        if len(self._root) == 0 and len(root) > 0:
            self._root = os.sep    # In case of the system root dir in linux
        self._suffix = suffix

        self._keys = dict(log=log, out=out, weight=weight, **subs)
        for k, v in self._keys.items():
            v_ = v.rstrip(os.sep)
            if len(v_) == 0 or not self.check_path(v_):
                _logger.warn("{} is not a valid path.".format(v))
                continue
            self._keys[k] = v_

        self._dir_tree = _Tree(
            self._root, 'root',
            eles=dict(zip(self._keys.values(), self._keys.keys())),
            sep=os.sep, none_val=''
        )

        self.add_keys(False)
        self.update_vfs(False)

        self.__counter = 0

    def __str__(self):
        return '\n'+self.sub_dirs

    @property
    def sub_dirs(self):
        return str(self._dir_tree)

    @property
    def root(self):
        return self._root

    def _add_key(self, key, val):
        self._keys.setdefault(key, val)

    def add_keys(self, verbose=False):
        for k, v in self._keys.items():
            self._add_key(k, v)
        if verbose:
            _logger.show(self._keys)
        
    def update_vfs(self, verbose=False):
        self._dir_tree.perform(lambda x: self.make_dir(x.path))
        if verbose:
            _logger.show("\nFolder structure:")
            _logger.show(self._dir_tree)

    @staticmethod
    def check_path(path):
        # This is to prevent stuff like A/../B or A/./.././C.d
        # Note that paths like A.B/.C/D are not supported, either.
        return osp.dirname(path).find('.') == -1

    @staticmethod
    def make_dir(path):
        if not osp.exists(path):
            os.mkdir(path)
        elif not osp.isdir(path):
            raise RuntimeError("Cannot create directory.")

    def get_dir(self, key):
        return osp.join(self.root, self._keys[key])

    def get_path(
        self, key, file, 
        name='', auto_make=False, 
        suffix=False, underline=True
    ):
        if len(file) == 0:
            return self.get_dir(key)
        if not self.check_path(file):
            raise ValueError("{} is not a valid path.".format(file))
        folder = self._keys[key]
        if suffix:
            path = osp.join(folder, self._add_suffix(file, underline=underline))
        else:
            path = osp.join(folder, file)

        if auto_make:
            base_dir = osp.dirname(path)
            # O(n) search for base_dir
            # Never update an existing key!
            if base_dir in self:
                _logger.warn("Cannot assign a new key to an existing path.")
                return osp.join(self.root, path)
            node = self._dir_tree.get_node(base_dir, mode='path')
            
            # Note that if name is an empty string,
            # the directory tree will be updated, but the name will not be added into self._keys.
            if node is None or node.is_placeholder():
                # Update directory tree
                des, visit = self._dir_tree.add_node(base_dir, name)
                # Create directories along the visiting path
                for d in visit: self.make_dir(d.path)
                self.make_dir(des.path)
            else:
                node.val = name
            if len(name) > 0:
                # Add new key
                self._add_key(name, base_dir)
        return osp.join(self.root, path)

    def _add_suffix(self, path, suffix='', underline=False):
        pos = path.rfind('.')
        if pos == -1:
            pos = len(path)
        _suffix = self._suffix if len(suffix) == 0 else suffix
        return path[:pos] + ('_' if underline and _suffix else '') + _suffix + path[pos:]

    def __contains__(self, value):
        return value in self._keys.values() or value == self._root

    def contains_key(self, key):
        return key in self._keys


class Registry(dict):
    def register(self, key, val):
        if key in self: _logger.warn("Key {} has already been registered.".format(key))
        self[key] = val
    
    def register_func(self, key):
        def _wrapper(func):
            self.register(key, func)
            return func
        return _wrapper


# Registry for global objects
R = Registry()
R.register('Logger', _logger)
register = R.register

# Registries for builders
MODELS = Registry()
OPTIMS = Registry()
CRITNS = Registry()
DATA = Registry()