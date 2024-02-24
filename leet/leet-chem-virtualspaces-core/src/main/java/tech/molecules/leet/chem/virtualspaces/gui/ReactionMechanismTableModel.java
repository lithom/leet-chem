package tech.molecules.leet.chem.virtualspaces.gui;

import com.actelion.research.chem.reaction.Reaction;

import javax.swing.table.AbstractTableModel;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ReactionMechanismTableModel extends AbstractTableModel {
    private final List<ReactionMechanism> reactionMechanisms;

    private Set<ReactionMechanism> selectedReactionMechanisms = new HashSet<>();
    private final String[] columnNames = {"Filepath", "Reaction", "Selected"};

    public ReactionMechanismTableModel(List<ReactionMechanism> reactionMechanisms) {
        this.reactionMechanisms = reactionMechanisms;
    }

    public void addReactionMechanism(ReactionMechanism rxn) {
        this.reactionMechanisms.add(rxn);
        this.fireTableRowsInserted(this.reactionMechanisms.size()-2,this.reactionMechanisms.size()-1);
    }

    @Override
    public int getRowCount() {
        return reactionMechanisms.size();
    }

    @Override
    public int getColumnCount() {
        return columnNames.length;
    }

    @Override
    public boolean isCellEditable(int rowIndex, int columnIndex) {
        if(columnIndex==2) {return true;}
        return false;
    }

    @Override
    public Class<?> getColumnClass(int columnIndex) {
        switch(columnIndex){
            case 0: return String.class;
            case 1: return Reaction.class;
            case 2: return Boolean.class;
        }
        return null;
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        ReactionMechanism mechanism = reactionMechanisms.get(rowIndex);
        switch (columnIndex) {
            case 0: return mechanism.filepath;
            case 1: return mechanism.rxn; // Assuming Reaction has a meaningful toString()
            case 2: return this.selectedReactionMechanisms.contains(mechanism);
            default: throw new IllegalArgumentException("Invalid column index");
        }
    }

    @Override
    public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
        if(columnIndex==2) {
            Boolean val = (Boolean) aValue;
            if(val) {this.selectedReactionMechanisms.add(this.reactionMechanisms.get(rowIndex));}
            else {this.selectedReactionMechanisms.remove(this.reactionMechanisms.get(rowIndex));}
        }
        fireTableRowsUpdated(rowIndex,rowIndex);
    }

    @Override
    public String getColumnName(int column) {
        return columnNames[column];
    }

    // Methods to add and remove ReactionMechanism objects similar to BuildingBlockFileTableModel
}
