# Run multiple run.sh script
args=($@)
project=${args[0]}
cd $project
for run_id in ${args[@]:1}
do
	pushd .
	cd $run_id
	bash run.sh |& tee -a out.log
	popd
done
