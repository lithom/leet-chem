package tech.molecules.leet.util;

import net.mahdilamb.colormap.Colormap;
import net.mahdilamb.colormap.Colormaps;
import net.mahdilamb.colormap.FluidColormap;
import org.jfree.chart.renderer.PaintScale;
import org.jfree.chart.renderer.xy.XYShapeRenderer;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.NumericalDatasource;
import tech.molecules.leet.table.chart.ScatterPlotModel;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

public class ColorMapHelper {


    public static Map<String,Color> evaluateColorValues(Colormap cm, NexusTableModel ntm, NumericalDatasource nd) {
        Map<String,Double> vi = new HashMap<>();
        for(String ri : ntm.getVisibleRows()) {
            if(nd.hasValue(ri)) {
                vi.put(ri,nd.getValue(ri));
            }
        }
        return evaluateColorValues(cm,vi);
    }

    public static Map<String,Color> evaluateColorValues(Colormap cm, Map<String,Double> val) {
        double c_min = Double.POSITIVE_INFINITY;
        double c_max = Double.NEGATIVE_INFINITY;
        //Random ri = new Random();
        //for(String ci : col.keySet()) { // no, we should color all rows, not assigned ones get a nan..
        for (String ci : val.keySet()) {
            Double cvi = val.get(ci);
            if (cvi == null) {
                cvi = Double.NaN;
            }
            //System.out.println("v: " + cvi);
            //this.dataCol.setValue(ci, cvi);
            if (Double.isFinite(cvi)) {
                c_min = Math.min(cvi, c_min);
                c_max = Math.max(cvi, c_max);
            }

        }

        double paintscale_min = c_min - Math.max(0.001, (c_max - c_min) * 0.01);
        double paintscale_max = c_max + Math.max(0.001, (c_max - c_min) * 0.01);
        //((XYShapeRenderer)this.cp.getChart().getXYPlot().getRenderer()).setPaintScale(new SpectrumPaintScale(paintscale_min,paintscale_max));

        if(cm == null) {
            cm = net.mahdilamb.colormap.Colormaps.get("Jet");
        }
        FluidColormap fcm = Colormaps.fluidColormap(cm);
        fcm.setMinValue((float)paintscale_min);
        fcm.setMaxValue((float)paintscale_max);

        Map<String,Color> cv = new HashMap<>();
        for(String ci : val.keySet()) {
            if(!Double.isNaN(val.get(ci))) {
                cv.put(ci, fcm.get(val.get(ci)));
            }
        }
        return cv;
        //PaintScaleFromColormap psfc = new PaintScaleFromColormap(cm, paintscale_min, paintscale_max, 0.75, new Color(180, 200, 210, 80));
        //((XYShapeRenderer)this.cp.getChart().getXYPlot().getRenderer()).setPaintScale(psfc);
    }

    public static Color createRandomColor() {
        double ri = Math.random();
        return net.mahdilamb.colormap.Colormaps.get("Plotly").get(ri);
    }


    public static class PaintScaleFromColormap implements PaintScale {

        private Colormap cm;
        private double lb, ub;
        double transparency;
        Color nanColor;

        public PaintScaleFromColormap(Colormap cm, double lb, double ub, double transparency, Color nanColor) {
            this.cm = cm;
            this.lb = lb; this.ub = ub;
            this.transparency = transparency;
            this.nanColor = nanColor;
        }

        @Override
        public double getLowerBound() {
            return this.lb;
        }

        @Override
        public double getUpperBound() {
            return this.ub;
        }

        @Override
        public Paint getPaint(double v) {
            if(Double.isNaN(v)) {
                if(this.nanColor!=null) {
                    return this.nanColor;
                    //return new Color( this.nanColor.getRed() , this.nanColor.getGreen() , this.nanColor.getBlue() , (int)(255*this.transparency) )
                }
            }
            double va = (v-lb)/(ub-lb);
            //return cm.get( va );
            Color ca = cm.get( va );
            return new Color(ca.getRed(),ca.getGreen(),ca.getBlue(), (int)(255*this.transparency) );
        }
    }

    public static class SpectrumPaintScale implements PaintScale {

        private static final float H1 = 0.25f;
        private static final float H2 = 0.75f;
        private final double lowerBound;
        private final double upperBound;

        public SpectrumPaintScale(double lowerBound, double upperBound) {
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
        }

        @Override
        public double getLowerBound() {
            return lowerBound;
        }

        @Override
        public double getUpperBound() {
            return upperBound;
        }

        @Override
        public Paint getPaint(double value) {
            float scaledValue = (float) (value / (getUpperBound() - getLowerBound()));
            float scaledH = H1 + scaledValue * (H2 - H1);
            return Color.getHSBColor(scaledH, 1f, 1f);
        }
    }

}
