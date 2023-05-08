package tech.molecules.chem.coredb;

import java.util.Date;

public interface AssayResult {

    public long  getId();
    public Assay getAssay();
    public Date getDate();
    public Tube getTube();

    public DataValue getData(AssayParameter ap);
    public DataValue getData(String parameter_name);
}


