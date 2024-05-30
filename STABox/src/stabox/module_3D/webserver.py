#!/usr/bin/env python3
import sys
import getopt
import json
from http.server import *
#####################################################
# Usage
#
def webserver_usage():
    print("""
Usage : vt3d AtlasBrowser LaunchAtlas [options]

Options:
            -p [port, default 80]
Example:
        > vt3d WebServer 
        
        ...
        never stop until you press Ctrl-C
""", flush=True)

class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)
#####################################################
# main pipe
#
def webserver_main(ports):
    #######################################
    # default parameter value
    # port = 8050

    # try:
    #     opts, args = getopt.getopt(argv,"hp:",["help"])
    # except getopt.GetoptError:
    #     webserver_usage()
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt in ('-h' ,'--help'):
    #         webserver_usage()
    #         sys.exit(0)
    #     elif opt in ("-p"):
    #         port = int(arg)

    # sanity check
    # run server
    print(f'server run in port {ports} now ...')
    server_address = ('', ports)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    httpd.serve_forever()

# if __name__ == '__main__':
#     # webserver_main(ports=80)
#     import os
#     # base = os.path.dirname(os.path.realpath(__file__))
#     # # print(sys.argv[0])
#     # os.chdir('D:/Users/lqlu/download/3D_SRT_Data/Stereo-seq-L1/webcache')
#     # # print(os.path.dirname(os.path.realpath(__file__)))
#     # webserver_main(ports=8050)
#     # http://127.0.0.1:8050/
#     # run Starmap
#     # base = os.path.dirname(os.path.realpath(__file__))
#     # # print(sys.argv[0])
#     # os.chdir('D:/Users/lqlu/download/3D_SRT_Data/STARMap/webcache')
#     # # print(os.path.dirname(os.path.realpath(__file__)))
#     # webserver_main(ports=8050)
#     # run DLPFC
#     # base = os.path.dirname(os.path.realpath(__file__))
#     # print(sys.argv[0])
#     # Run DLPFC
#     # os.chdir('D:/Users/lqlu/download/3D_SRT_Data/DLPFC/webcache')
#     # webserver_main(ports=8050)
#     # Run SlideSeqV2
#     # os.chdir('D:/Users/lqlu/download/3D_SRT_Data/SlideSeqV2/webcache')
#     # webserver_main(ports=8050)
#     # Run Multi-DLPFC
#     os.chdir('D:/Users/lqlu/download/3D_SRT_Data/Multi-DLPFC/webcache')
#     webserver_main(ports=8050)
