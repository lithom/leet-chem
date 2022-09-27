package tech.molecules.leet.table;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.clustering.ClusterAppModel;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

/**
 *
 * @param <T> data provider
 */
public interface NClassification {
    public static interface NClass {
        public String getName();
        public String getDescription();
        public Color getColor();
        public void setColor(Color c);
        public boolean isMember(String ki);
        public Set<String> getMembers();
    }

    public static interface NClassWithTimestamp {
        public Date getTimestamp(String ki);
    }

    public List<NClass> getClasses();
    public void addClass(NClass ci);
    public void removeClass(NClass ci);

    public interface ClassificationListener {
        public void classificationChanged();
        public void classChanged(NClass ci);
    }

    public void addClassificationListener(ClassificationListener li);
    public void removeClassificationListener(ClassificationListener li);


    public List<NClassification.NClass> getClassesForRow(String rowid);



    /**
     * Helper class that provides a table model showing the classes of a classification.
     *
     */
    public static class ClassificationTableModel extends AbstractTableModel {

        private NClassification nc;
        //private List<NClass> clusters;

        public ClassificationTableModel(NClassification nc) {
            this.nc = nc;
            //this.clusters = nc.getClasses();
            nc.addClassificationListener(new ClassificationListener() {
                @Override
                public void classificationChanged() {
                    fireClustersChanged();
                }

                @Override
                public void classChanged(NClass ci) {
                    fireClustersChanged();
                }
            });
        }

        @Override
        public String getColumnName(int column) {
            switch(column) {
                case 0: return "Name";
                case 1: return "Size";
                case 2: return "Color";
            }
            return "<ERROR>";
        }

        @Override
        public int getRowCount() {
            return nc.getClasses().size();
        }

        @Override
        public int getColumnCount() {
            return 3;
        }

        @Override
        public Object getValueAt(int rowIndex, int columnIndex) {
            if(rowIndex>=nc.getClasses().size()){return null;}
            NClass ci = nc.getClasses().get(rowIndex);
            switch(columnIndex){
                case 0: return ci.getName();
                case 1: return ci.getMembers().size();
                case 2: return ci.getColor();
            }
            return null;
        }

        public void fireClustersChanged() {
            //this.fireTableStructureChanged();
            this.fireTableDataChanged();
        }
    }
}
