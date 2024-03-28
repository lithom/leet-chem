package tech.molecules.leet.chem.virtualspaces.gui;

import com.actelion.research.chem.reaction.Reaction;

import java.io.File;
import java.util.ArrayList;

public class SpaceCreatorController {

    private SpaceCreatorView view;

    private SpaceCreatorModel model;

    public SpaceCreatorController() {
        initModel();
        initView();

    }

    private void initModel() {
        BuildingBlockFileTableModel buildingBlockFileTableModel = new BuildingBlockFileTableModel(new ArrayList<>());
        ReactionMechanismTableModel reactionMechanismTableModel = new ReactionMechanismTableModel(new ArrayList<>());
        this.model = new SpaceCreatorModel(buildingBlockFileTableModel,reactionMechanismTableModel);
    }

    private void initView() {
        // Initialize the view and pass the table models
        view = new SpaceCreatorView(this,this.model);
        // Add listeners to the view for actions like adding BuildingBlockFiles, etc.
    }

    public SpaceCreatorView getView() {
        return view;
    }

    public SpaceCreatorModel getModel() {
        return model;
    }

    // Methods to handle actions from the view, like adding a BuildingBlockFile
    public void addBuildingBlockFile(BuildingBlockFile file) {
        this.model.getBuildingBlockFileTableModel().addBuildingBlockFile(file);
        BuildingBlockFileAnalysisWorker worker = new BuildingBlockFileAnalysisWorker(this.model.getBuildingBlockFileTableModel(),file);
        worker.execute();
    }

    public void addReaction(File rxnFile) throws Exception {
        this.model.getReactionMechanismTableModel().addReactionMechanism(new ReactionMechanism(rxnFile.getPath()));
    }

}
