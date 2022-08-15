package tech.molecules.leet.table.action;

import com.actelion.research.gui.VerticalFlowLayout;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;
import tech.molecules.leet.table.chart.JFreeChartScatterPlot;
import tech.molecules.leet.table.gui.JNumericalDataSourceSelector;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.function.Supplier;



public class CreateXYScatterPlotAction extends AbstractAction {

    private NexusTableModel ntm;
    private Supplier<JPanel> plotPanel;
    private Frame owner;

    private CreateXYScatterPlotActionModel model;
    private CreateXYScatterPlotActionGUI gui;

    /**
     *
     * @param ntm
     * @param plotPanel the .get() function will be called exactly once to get the panel
     * @param owner
     */
    public CreateXYScatterPlotAction(NexusTableModel ntm, Supplier<JPanel> plotPanel, Frame owner) {
        super("Create XY Plot");
        this.ntm = ntm;
        this.plotPanel = plotPanel;
        this.owner = owner;
    }

    @Override
    public void actionPerformed(ActionEvent e) {

        JPanel targetPanel = plotPanel.get();
        try {
            this.model = new CreateXYScatterPlotActionModel(ntm,targetPanel);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this.owner,ex.getMessage());
        }
        this.gui = new CreateXYScatterPlotActionGUI(this.model,owner);


        //JFreeChartScatterPlot csp = new JFreeChartScatterPlot();
        JDialog jd = new JDialog(owner,true);
        jd.getContentPane().setLayout(new BorderLayout());
        jd.getContentPane().add( this.gui.getActionGUI() , BorderLayout.CENTER);

        JPanel bottom = new JPanel();
        bottom.setLayout(new FlowLayout(FlowLayout.RIGHT));
        JButton jb_close = new JButton("OK");
        bottom.add(jb_close);
        jb_close.addActionListener( (actionEvent) -> jd.dispose() );
        jd.getContentPane().add(bottom,BorderLayout.SOUTH);
        jd.pack();
        jd.setVisible(true);

        this.model.setConfig( this.gui.getConfig(this.model.getNTM(),targetPanel) );
        this.model.performAction();
    }

}
