import argparse
from HVAC_RL_web_worker_client.web_worker_server import WorkerServer

def main(): 
    parser = argparse.ArgumentParser(description='Run WorkerServer')
    parser.add_argument('--state_syn_interval', default=5, type=int, help='Interval to sync states, in seconds.')
    parser.add_argument('--ip', type=str, help='The binding IP.')
    args = parser.parse_args();

    worker_server_ins = WorkerServer(args.state_syn_interval, args.ip);
    worker_server_ins.runWorkerServer();


if __name__ == '__main__':
    main()
