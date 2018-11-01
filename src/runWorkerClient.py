import argparse
from HVAC_RL_web_worker_client.web_worker_client import WorkerClient


def main(): 
    parser = argparse.ArgumentParser(description='Run WorkerClient')
    parser.add_argument('--port', default=7777, type=int, help='The port this client is binding.')
    parser.add_argument('--max_work_num', default=2, type=int, help='The maximum number of experiments to run at same time.')
    args = parser.parse_args();

    worker_client_ins = WorkerClient(args.port, args.max_work_num);
    worker_client_ins.runWorkerClient();


if __name__ == '__main__':
    main()
