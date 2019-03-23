"""
This script reads the eval results of the exp run.
"""
import argparse, os

def main():
    parser = argparse.ArgumentParser(description='Get exp run results.')
    parser.add_argument('--prj_part_id', type=str, help='The project part ID')
    parser.add_argument('--prj_location', type=str, help='The project location (e.g. pit, bej, shg, sgp)')
    parser.add_argument('--prj_weight', type=str, help='The project weight type')
    parser.add_argument('--prj_run_id', type=str, help='The project run id')
    parser.add_argument('--res_batch', type=str, help='The exp results batch number')
    parser.add_argument('--res_id', type=str, help='The exp results id')
    args = parser.parse_args();
    read_run_res(args.prj_part_id, args.prj_location, args.prj_weight, 
                args.prj_run_id, args.res_batch, args.res_id);
        
def read_run_res(part_id, location, weight, run_id, res_batch, res_id):
    ret = [];
    prj_path = './../runs/rl_parametric_part%s_%s_%s/%s/'%(part_id, location, weight, run_id);
    for test_i in range(1, 5):
        eval_res_path = prj_path + 'Eplus-env-Part%s-%s-%s-Test-v%s-res%s/Eplus-env-sub_run%s'%(part_id, 
                        weight.title(), location.title(), test_i, res_batch, res_id);
        eval_res_html_path = eval_res_path + '/output/eplustbl.htm'
        ret.append(get_res_from_html(eval_res_html_path));
        os.rename(eval_res_path, eval_res_path + '*');
    eval_res_path = prj_path + 'Eplus-env-Part%s-%s-%s-Train-v%s-res%s/Eplus-env-sub_run%s'%(part_id, 
                        weight.title(), location.title(), 1, res_batch, res_id);
    eval_res_html_path = eval_res_path + '/output/eplustbl.htm';
    ret.append(get_res_from_html(eval_res_html_path));
    os.rename(eval_res_path, eval_res_path + '*');
    print (ret)


def get_res_from_html(eval_res_path):
    ret = [];
    with open(eval_res_path, 'r') as html_f:
        html_lines = html_f.readlines();
        to_read_flag = -1;
        for line in html_lines:
            if to_read_flag != 0:
                if 'Total Site Energy' in line:
                    to_read_flag = 3;
                if 'Number of hours heating loads not met' in line:
                    to_read_flag = 1;
                if 'Number of hours cooling loads not met' in line:
                    to_read_flag = 1;
            else:
                res_i = line.split('>')[1].strip().split('<')[0].strip();
                ret.append(res_i);
            to_read_flag -= 1;
    return ret;



if __name__ == '__main__':
    main()
