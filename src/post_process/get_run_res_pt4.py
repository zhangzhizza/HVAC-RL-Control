"""
This script reads the eval results of the exp run.
"""
import argparse, os

def main():
    parser = argparse.ArgumentParser(description='Get exp run results.')
    parser.add_argument('--prj_location', type=str, help='The project location (e.g. pit, bej, shg, sgp)')
    parser.add_argument('--prj_weight', type=str, help='The project weight type')
    parser.add_argument('--prj_run_id', type=str, help='The project run id')
    parser.add_argument('--res_batch', type=str, help='The exp results batch number')
    parser.add_argument('--res_id', type=str, help='The exp results id')
    parser.add_argument('--test_init_i', type=int, help='The test env initial version')
    parser.add_argument('--train_i', type=int, help='The train env initial version')
    parser.add_argument('--repeat_flag', type=bool, help='The repeat version flag')
    args = parser.parse_args();
    read_run_res(4, args.prj_location, args.prj_weight, 
                args.prj_run_id, args.res_batch, args.res_id,
                args.test_init_i, args.train_i, args.repeat_flag);
        
def read_run_res(part_id, location, weight, run_id, res_batch, res_id, test_init_i, train_i,
                repeat_flag):
    ret = [];
    prj_path = './../runs/rl_parametric_part%s_%s_%s/%s/'%(part_id, location, weight, run_id);
    for test_i in range(test_init_i, test_init_i+2):
        if not repeat_flag:
            eval_res_path = prj_path + 'Eplus-env-Part%s-%s-%s-Test-v%s-res%s/Eplus-env-sub_run%s'%(part_id, 
                            weight.title(), location.title(), test_i, res_batch, res_id);
            eval_res_eso_path = eval_res_path + '/output/eplusout.eso'
            ret.append(get_res_from_eso(eval_res_eso_path));
        else:
            eval_res_path = prj_path + 'Eplus-env-Part%s-%s-%s-Test-Repeat-v%s-res%s/Eplus-env-sub_run%s'%(part_id, 
                            weight.title(), location.title(), test_i, res_batch, res_id);
            eval_res_eso_path = eval_res_path + '/output/eplusout.eso'
            ret.append(get_res_from_eso(eval_res_eso_path));
        os.rename(eval_res_path, eval_res_path + '*');

    if not repeat_flag:
        eval_res_path = prj_path + 'Eplus-env-Part%s-%s-%s-Train-v%s-res%s/Eplus-env-sub_run%s'%(part_id, 
                        weight.title(), location.title(), train_i, res_batch, res_id);
    else:
        eval_res_path = prj_path + 'Eplus-env-Part%s-%s-%s-Train-Repeat-v%s-res%s/Eplus-env-sub_run%s'%(part_id, 
                        weight.title(), location.title(), train_i, res_batch, res_id);
    eval_res_eso_path = eval_res_path + '/output/eplusout.eso';
    ret.append(get_res_from_eso(eval_res_eso_path));
    os.rename(eval_res_path, eval_res_path + '*');
    print (ret)


def get_res_from_eso(eval_res_path):
    with open(eval_res_path, 'r') as eso_f:
        eso_lines = eso_f.readlines();
        header_idx_dict = {};
        is_reading_header = True;
        avg_enrg = round(float(eso_lines[152959-1].split(',')[1]), 0)
        tol_enrg = avg_enrg;
        not_met = float(eso_lines[152958-1].split(',')[1])
        avg_pmv = round(float(eso_lines[152947-1].split(',')[1]), 4)
    ret = [tol_enrg, not_met, avg_pmv];        
    return ret;



if __name__ == '__main__':
    main()
