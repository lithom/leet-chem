package tech.molecules.chem.coredb.sql;

import tech.molecules.chem.coredb.Batch;
import tech.molecules.chem.coredb.Compound;

public class BatchImpl implements Batch {
    private String id;
    private String compoundId;

    public BatchImpl(String id, String compoundId) {
        this.id = id;
        this.compoundId = compoundId;
    }

    public String getId() { return id; }
    public String getCompoundId() { return compoundId; }
}
