import yaml
import os
import json
import re

# Point this to the template. This contains everything to be populated in between three sets of dashes (e.g. ---n---)
template='/home/ftijhuis/software/dnfitting/submissions/template_2folds.sh'
# Point this to the yaml This contains all the things that are going to be the same for each job (i.e. the email, the requested resources etc).
myyaml='/home/ftijhuis/software/dnfitting/submissions/examp_yaml.yml'
# point this to where you want the jobscripts output
out_dir='/home/ftijhuis/JOBS/'


# Define a class for populating template sh file given a yaml file
class Script_Populator:

    """Script_Populator
    Class for populating a script.

    The idea is to take a template file and populate it with information contained in a yaml file.


    """


    def __init__(self,yaml_file,template_file,out_dir,jobname='myfitting',suffix='.sh',**kwargs):

        """
        Initialise the class.

        Parameters
        ----------
        yaml_file: yaml file containing the information to be fed into the jobscript.
        template_file: template file for the jobscript
        out_dir: Where to save the populated script.
        jobname: The name given to the job.


        An additional 'supdict' dictionary can be provided in kwargs to populate additional information.
        This is useful in the case where the script needs to be populated on the fly.

        Parameters
        ----------

        self.outfile: Script output location.
        self.working_string: The unpopulated template script.

        """
        self.jobname=jobname # Name to be given to job.

        self.yaml_file=yaml_file


        with open(yaml_file, 'r') as f:
            self.yaml = yaml.safe_load(f)

        # Append the supdict if it exists.
        if 'supdict' in kwargs:
            supdict = kwargs.get('supdict')
            self.yaml={**self.yaml, **supdict}

        # Read the jobscript template into memory.
        self.jobscript = open(template_file)
        self.working_string = self.jobscript.read()
        self.jobscript.close()

        subject = supdict['---subject---']
        slice_n = supdict['---data_portion---']
        session = supdict['---session---']

        self.outfile=os.path.join(out_dir, f'sub-{subject}_ses-{session}_slice-{slice_n}' + suffix)


    def populate(self):

        """ populate

        Populates the jobscript with items from the yaml

        Returns
        ----------
        self.working_string: populated script.

        """


        for e in self.yaml:
            rS = re.compile(e)
            self.working_string = re.sub(rS, self.yaml[e], self.working_string)

    def writeout(self):

        """  writeout

        Writes out the jobscript file to the outfile location.

        """


        of = open(self.outfile, 'w')
        of.write(self.working_string)
        of.close()

    def execute(self,execute_type='sbatch'):

        """ execute
        Executes the script.
        """

        os.system(execute_type + " " + self.outfile)
        print('{job} sent to SLURM'.format(job=self.jobname))

# Create a list of dictionaries of arguments that need to be populated dynamically.
# Here we create a dictionary for every instance of subject and data portion
mysuppdicts=[dict({'---subject---':str(p),'---data_portion---':str(dp), '---session---':str(ses)}) for p in ['015', '016'] for dp in range(20) for ses in [2,3]] #357

# Now we run through a big loop that populates the template script based on everything in each dictionary and writes it to a file.
#You can uncomment the last line to send them all to slurm (check first though).
for cdict in mysuppdicts:
    x=Script_Populator(myyaml,template,out_dir,jobname=json.dumps(cdict),supdict=cdict)
    x.populate()
    x.writeout()
    # Only uncomment the next line after checking the files have populated correctly
    x.execute()
