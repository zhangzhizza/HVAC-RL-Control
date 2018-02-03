for dir in ${1}*/
do
	pushd .
	cd ${dir}output
	../../../../../eplus-env/eplus_env/envs/EnergyPlus-8-3-0/ReadVarsESO
	popd
done 