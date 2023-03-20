package tech.molecules.chem.coredb.medchem;

import tech.molecules.chem.coredb.Compound;

import java.util.Date;

public interface SeriesMember {
    public Compound getCompound();
    public Date getDateAdded();
}
