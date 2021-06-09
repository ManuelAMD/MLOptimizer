from app.init_nodes import InitNodes
from app.common.dataset import *
from app.common.search_space import *

if __name__ == '__main__':
    masterNode = InitNodes()
    masterNode.master()