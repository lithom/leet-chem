package tech.molecules.chem.coredb.aggregation;

import tech.molecules.chem.coredb.AssayResult;

interface ResultFilter {

    public boolean filter(AssayResult ri);

}
