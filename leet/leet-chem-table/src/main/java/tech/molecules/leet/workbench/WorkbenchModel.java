package tech.molecules.leet.workbench;

import tech.molecules.leet.table.NexusTableModel;

public class WorkbenchModel {

    private NexusTableModel ntm;

    public WorkbenchModel(NexusTableModel ntm) {
        this.ntm = ntm;
    }

    public NexusTableModel getNexusTableModel() {
        return this.ntm;
    }

}
