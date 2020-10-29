"""The SQLAlchemy models we use to persist data.
"""

import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)




class Pipeline():

    def __init__(self, origin='export', dataset='dataset'):
        self.id = uuid.uuid4()
        self.origin = origin
        self.dataset = dataset
        self.parameters = []
        self.modules = []
        self.connections = []
        self.created_date = datetime.now()

    def add_module(self, module):
        self.modules.append(module)

    def add_parameters(self,parameters):
        self.parameters.append(parameters)

    def add_connection(self, connection):
        self.connections.append(connection)


    def __eq__(self, other):
        return type(other) is Pipeline and other.id == self.id

    def __repr__(self):
        return '<Pipeline %r>' % self.id

    __str__ = __repr__


class PipelineModule():
    def __init__(self, package, version, name):
        self.id = uuid.uuid4()
        self.package = package
        self.version = version
        self.name = name
        self.connections_to = []

    def add_connection_to(self, connection):
        self.connections_to.append(connection)

class PipelineConnection():
    def __init__(self, from_module_id, from_output_name, to_module_id, to_input_name):
        self.from_module_id = from_module_id
        self.from_output_name = from_output_name
        self.to_module_id = to_module_id
        self.to_input_name = to_input_name



class PipelineParameter():
    def __init__(self,module_id, name, value):
        self.module_id = module_id
        self.name = name
        self.value = value


