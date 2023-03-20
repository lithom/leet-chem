package tech.molecules.chem.coredb.medchem;

import tech.molecules.chem.coredb.Project;

import java.util.List;
import java.util.Set;

public interface Series {
    public String getName();
    public Set<SeriesMember> getMembers();
    public Set<SeriesMember> getDefiningMembers();
}
