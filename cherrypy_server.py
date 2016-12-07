import evospace
import cherrypy

from cherrypy._cpcompat import ntou

import json, os


class Content:
    def __init__(self, popName="pop"):
        self.population = evospace.Population(popName)
        self.population.initialize()

    @cherrypy.expose
    @cherrypy.tools.json_in(content_type=[ntou('application/json'),
                                          ntou('text/javascript'),
                                          ntou('application/json-rpc')
                                          ])
    def evospace(self):
        if cherrypy.request.json:
            obj = cherrypy.request.json
            method = obj["method"]
            _id = obj["id"]

            if "params" in obj:
                params = obj["params"]
            else:
                return json.dumps({"result": None, "error":
                    {"code": -32604, "message": "Params empty"}, "id": _id})

            # process the data
            cherrypy.response.headers['Content-Type'] = 'text/json-comment-filtered'
            result = None
            if method == "initialize":
                result = self.population.initialize()
                return json.dumps({"result": result, "error": None, "id": _id})

            if method == "getSample":
                result = self.population.get_sample(params[0])
                if result:
                    return json.dumps({"result": result, "error": None, "id": _id})
                else:
                    return json.dumps({"result": None, "error":
                        {"code": -32601, "message": "EvoSpace empty"}, "id": _id})
            elif method == "respawn":
                result = self.population.respawn(params[0])
            elif method == "putSample":
                result = self.population.put_sample(params[0])
            elif method == "putIndividual":
                result = self.population.put_individual(**params[0])
            elif method == "size":
                result = self.population.size()
            elif method == "found":
                result = self.population.found()
            elif method == "found_it":
                result = self.population.found_it()

            return json.dumps({"result": result, "error": None, "id": _id})

        else:
            print "blah"
            return "blah"

    @cherrypy.expose
    def index(self):
        return "Servidor Funcionando"


if __name__ == '__main__':
    cherrypy.config.update({'server.socket_host': '0.0.0.0',
                            'server.socket_port': int(os.environ.get('PORT', '5000'))
                               , 'server.environment': 'production'
                               , 'server.thread_pool': 200
                               , 'tools.sessions.on': False
                               , 'server.socket_timeout': 30
                            })

    from cherrypy.process import servers


    def fake_wait_for_occupied_port(host, port):
        return


    servers.wait_for_occupied_port = fake_wait_for_occupied_port

    cherrypy.quickstart(Content('pop'))