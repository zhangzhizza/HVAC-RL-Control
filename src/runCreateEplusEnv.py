import argparse, os
import eplus_env_util.eplus_env_creator as creator

def get_abs_path(rel_path):
	return_path = rel_path;
	if rel_path[0] != '/':
		return_path = os.getcwd() + '/' + rel_path;
	return return_path;


def main(): 
    parser = argparse.ArgumentParser(description='Run EplusEnvCreator')
    parser.add_argument('--base_idf_path', type=str, help='The base IDF file path.')
    parser.add_argument('--add_idf_path', type=str, help='The add IDF file path.')
    parser.add_argument('--cfg_path', type=str, help='The cfg file path.')
    parser.add_argument('--env_name', type=str, help='The new Gym env name.')
    parser.add_argument('--weather_path', type=str, help='The weather file path.')
    parser.add_argument('--state_limit_path', type=str, help='The state limit file path.')
    parser.add_argument('--ext_sch_paths', nargs='+', type=str, default = [],
    					help='The IDF external schedule files paths.')
    parser.add_argument('--eplus_version', type=str, help='The EnergyPlus engine version, e.g. 8-3-0')
    args = parser.parse_args();

    # Change to abs path if the input is relative path
    args.base_idf_path = get_abs_path(args.base_idf_path);
    args.add_idf_path = get_abs_path(args.add_idf_path);
    args.cfg_path = get_abs_path(args.cfg_path);
    args.weather_path = get_abs_path(args.weather_path);
    ext_sch_paths = [];
    for ext_sch_path in args.ext_sch_paths:
    	ext_sch_paths.append(get_abs_path(ext_sch_path));
    args.state_limit_path = get_abs_path(args.state_limit_path);
    args.ext_sch_paths = ext_sch_paths;

    print (args)

    env_creator = creator.EplusEnvCreator();
    env_creator.create_env(args.base_idf_path, args.add_idf_path, args.cfg_path, args.env_name,
    						args.weather_path, args.state_limit_path, args.ext_sch_paths,
                            args.eplus_version);


if __name__ == '__main__':
    main()
