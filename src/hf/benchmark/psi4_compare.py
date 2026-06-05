import psi4

for names in ["1-butanol.xyz", 
              "1-butene.xyz", 
              "1-butylamine.xyz", 
              "1-hexanol.xyz", 
              "1-hexene.xyz", 
              "1-hexylamine.xyz", 
              "1-pentanol.xyz", 
              "1-pentene.xyz", 
              "1-pentylamine.xyz", 
              "1-propanol.xyz", 
              "1-propene.xyz", 
              "1-propylamine.xyz", 
              "ammonia.xyz", 
              "butane.xyz", 
              "carbon_dioxide.xyz", 
              "carbon_monoxide.xyz", 
              "cyclobutane.xyz", 
              "cyclohexane.xyz", 
              "cyclopentane.xyz", 
              "cyclopropane.xyz", 
              "ethane.xyz", 
              "ethanol.xyz", 
              "ethene.xyz", 
              "ethylamine.xyz", 
              "furan.xyz", 
              "hexane.xyz", 
              "hydrogen_peroxide.xyz", 
              "hydrogen_sulfide.xyz", 
              "methane.xyz", 
              "methanol.xyz", 
              "methylamine.xyz", 
              "oxirane.xyz", 
              "pentane.xyz", 
              "propane.xyz", 
              "pyridine.xyz", 
              "pyrrole.xyz", 
              "water.xyz"]: 
    print(names, end='\t')
    ss = open('../data/molecules/'+names, 'r').read()
    ss = '\n'.join(ss.split('\n')[2:])
    mol = psi4.geometry(ss)
    psi4.set_options(
        {'basis': 'cc-pvdz', 
        'puream': False, 
        'scf_type': 'pk', 
        'reference': 'rhf', 
        'e_convergence': 1e-8, 
        'd_convergence': 1e-6}
    )

    psi4.core.set_output_file(names[:-4]+'.out', False)
    e, _ = psi4.energy('scf', return_wfn=True)
    print(e)
    print()