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
                        weight.upper(), location.title(), test_i, res_batch, res_id);
        eval_res_eso_path = eval_res_path + '/output/eplusout.eso'
        ret.append(get_res_from_eso(eval_res_eso_path));
        os.rename(eval_res_path, eval_res_path + '*');
    eval_res_path = prj_path + 'Eplus-env-Part%s-%s-%s-Train-v%s-res%s/Eplus-env-sub_run%s'%(part_id, 
                        weight.upper(), location.title(), 1, res_batch, res_id);
    eval_res_eso_path = eval_res_path + '/output/eplusout.eso';
    ret.append(get_res_from_eso(eval_res_eso_path));
    os.rename(eval_res_path, eval_res_path + '*');
    print (ret)


def get_res_from_eso(eval_res_path):
    with open(eval_res_path, 'r') as eso_f:
        eso_lines = eso_f.readlines();
        header_idx_dict = {};
        is_reading_header = True;
        sim_days = int(eso_lines[355545-1].split(',')[1])
        avg_enrg = float(eso_lines[355549-1].split(',')[1])
        tol_enrg = avg_enrg * sim_days * 24 / 1000.0;
        not_met = float(eso_lines[355546-1].split(',')[1])
        sht_cyc = float(eso_lines[355547-1].split(',')[1])
        low_plr = float(eso_lines[355548-1].split(',')[1])
    ret = [tol_enrg, not_met, sht_cyc, low_plr];        
    return ret;



if __name__ == '__main__':
    main()
