wget https://files.rcsb.org/download/1AKI.pdb
grep -v HOH 1AKI.pdb > 1AKI_clean.pdb
gmx pdb2gmx -f 1AKI_clean.pdb -o 1AKI_processed.gro -water spce
gmx editconf -f 1AKI_processed.gro -o 1AKI_newbox.gro -c -d 1.0 -bt cubic
gmx solvate -cp 1AKI_newbox.gro -cs spc216.gro -o 1AKI_solv.gro -p topol.top
wget http://www.mdtutorials.com/gmx/lysozyme/Files/ions.mdp
gmx grompp -f ions.mdp -c 1AKI_solv.gro -p topol.top -o ions.tpr
gmx genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral
wget http://www.mdtutorials.com/gmx/lysozyme/Files/minim.mdp
gmx grompp -f minim.mdp -c 1AKI_solv_ions.gro -p topol.top -o em.tpr
