package tech.molecules.leet.chem.virtualspaces.gui;

public class SpaceCreatorModel {

    private BuildingBlockFileTableModel buildingBlockFileTableModel;
    private ReactionMechanismTableModel reactionMechanismTableModel;

    public SpaceCreatorModel(BuildingBlockFileTableModel buildingBlockFileTableModel, ReactionMechanismTableModel reactionMechanismTableModel) {
        this.buildingBlockFileTableModel = buildingBlockFileTableModel;
        this.reactionMechanismTableModel = reactionMechanismTableModel;
    }

    public BuildingBlockFileTableModel getBuildingBlockFileTableModel() {
        return buildingBlockFileTableModel;
    }

    public void setBuildingBlockFileTableModel(BuildingBlockFileTableModel buildingBlockFileTableModel) {
        this.buildingBlockFileTableModel = buildingBlockFileTableModel;
    }

    public ReactionMechanismTableModel getReactionMechanismTableModel() {
        return reactionMechanismTableModel;
    }

    public void setReactionMechanismTableModel(ReactionMechanismTableModel reactionMechanismTableModel) {
        this.reactionMechanismTableModel = reactionMechanismTableModel;
    }
}
