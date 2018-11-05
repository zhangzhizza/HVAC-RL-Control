import argparse
from HVAC_RL_web_worker_client.web_worker_server import WorkerServer

def main(): 
    parser = argparse.ArgumentParser(description='Run WorkerServer')
    parser.add_argument('--state_syn_interval', default=5, type=int, help='Interval to sync states, in seconds.')
    parser.add_argument('--port', default=6666, type=int, help='The binding port.')
    args = parser.parse_args();

    worker_server_ins = WorkerServer(args.state_syn_interval, args.port);
    worker_server_ins.runWorkerServer();


if __name__ == '__main__':
    main()
