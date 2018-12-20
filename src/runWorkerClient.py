import argparse
from HVAC_RL_web_worker_client.web_worker_client import WorkerClient


def main(): 
    parser = argparse.ArgumentParser(description='Run WorkerClient')
    parser.add_argument('--max_work_num', default=2, type=int, help='The maximum number of experiments to run at same time.')
    parser.add_argument('--ip', type=str, help='The IP this client is binding.')
    args = parser.parse_args();

    worker_client_ins = WorkerClient(args.max_work_num, args.ip);
    worker_client_ins.runWorkerClient();


if __name__ == '__main__':
    main()
