package tech.molecules.leet;

import java.io.File;

public abstract class AbstractBatchJob {

    private File configurationFile;

    public void setConfigurationFile(File confFile) {
        this.configurationFile = confFile;
    }

    public abstract void runComputation();

}
