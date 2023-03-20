package tech.molecules.chem.coredb;

import java.util.List;

public interface Assay {
    public String getName();
    public int getId();
    public List<AssayParameter> getParameter();
}
