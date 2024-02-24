package tech.molecules.leet.chem.virtualspaces.gui;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.io.SDFileParser;
import org.apache.commons.lang3.tuple.Pair;

import javax.swing.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BuildingBlockFileAnalysisWorker extends SwingWorker<Void, Void> {
    private BuildingBlockFileTableModel tableModel;
    private BuildingBlockFile bbFile;
    private List<LoadedBB> loadedBBs = new ArrayList<>();
    public BuildingBlockFileAnalysisWorker(BuildingBlockFileTableModel buildingBlockFileTableModel, BuildingBlockFile file) {
        this.tableModel = buildingBlockFileTableModel;
        this.bbFile = file;
    }

    private void setTextFailed(String msg) {
        this.bbFile.setStatus("[X] "+msg);
        tableModel.updateStatus(this.bbFile);
    }
    private void setTextStatus(String msg) {
        this.bbFile.setStatus(msg);
        tableModel.updateStatus(this.bbFile);
    }

    private void setLoadedBBs() {
        this.tableModel.setLoadedBBs(this.bbFile,this.loadedBBs);
    }

    @Override
    protected Void doInBackground() throws Exception {
        // Your file analysis logic here
        //BuildingBlockFile file = buildingBlockFiles.get(rowIndex);
        System.out.println("Analyzing file: " + bbFile.getFilepath());

        if(bbFile.getFormat().equalsIgnoreCase(".sdf")) {
            String pathBBFile = this.bbFile.getFilepath();
            String idfieldName = this.bbFile.getIdFieldName();
            SDFileParser parser = new SDFileParser(pathBBFile);
            String[] columns = parser.getFieldNames();
            parser.close();
            parser = new SDFileParser(pathBBFile, columns);
            int idField = parser.getFieldIndex(idfieldName);
            if(idField < 0) {

            }
            int cnt = 0;
            while (parser.next()) {
                //if(cnt>=20000) {break;}
                String molid = parser.getFieldData(idField);
                StereoMolecule bb = parser.getMolecule();
                bb.ensureHelperArrays(Molecule.cHelperParities);
                this.loadedBBs.add(new LoadedBB(molid,bb.getIDCode(),bb));

                //bbs_1.add(Pair.of(bb.getIDCode(),bb.getAtoms()));
                //bbData.putIfAbsent(bb.getIDCode(), new HashMap<String, List<String>>());
                //Map<String, List<String>> propertyMap = bbData.get(bb.getIDCode());
                //propertyMap.putIfAbsent("BB-ID", new ArrayList<>());
                //propertyMap.get("BB-ID").add(enamineID);
                cnt++;
                //if(bbs.size()>2000) {break;}
                String status = "[Loading] "+this.loadedBBs.size();
                if(cnt%12000 == 0) {setTextStatus(  status + " ." );}
                else if(cnt%8000 == 0) {setTextStatus( status + " .." );}
                else if(cnt%4000 == 0) {setTextStatus( status + " ..." );}
            }
            setTextStatus("[Loaded] "+this.loadedBBs.size()+" BBs");
            setLoadedBBs();
        }
        else {
            setTextFailed("unsupported format");
        }

        return null;
    }

    @Override
    protected void done() {
        // This method is called when the background task is finished
        // Update your model or UI here if necessary
        try {
            get(); // Call get to catch any exceptions thrown by doInBackground()
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
