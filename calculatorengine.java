package com.example.calculator.engine;

import com.example.calculator.utils.NumberFormatter;

/**
 * Xử lý toàn bộ business logic của Basic Calculator.
 *
 * Engine không phụ thuộc Android UI.
 *
 * MainActivity chỉ gửi action:
 *
 * inputDigit()
 * inputDecimal()
 * inputOperator()
 * calculate()
 * clear()
 * backspace()
 * toggleSign()
 * percent()
 *
 * Sau đó đọc CalculatorState để update UI.
 */
public class CalculatorEngine {

    private final CalculatorState state;


    public CalculatorEngine() {
        state = new CalculatorState();
    }


    /**
     * Cho UI đọc trạng thái hiện tại.
     */
    public CalculatorState getState() {
        return state;
    }


    // =========================================================
    // NUMBER INPUT
    // =========================================================

    /**
     * Nhập một chữ số từ 0 đến 9.
     */
    public void inputDigit(String digit) {

        if (digit == null || !digit.matches("[0-9]")) {
            return;
        }

        /*
         * Nếu calculator đang Error:
         *
         * nhập số mới sẽ bắt đầu calculation mới.
         */
        if (state.isError()) {
            state.reset();
        }

        String current = state.getCurrentInput();

        /*
         * Nếu:
         *
         * - vừa chọn operator
         * - hoặc display hiện "0"
         *
         * thì digit mới thay thế display hiện tại.
         */
        if (state.isWaitingForOperand()
                || "0".equals(current)) {

            state.setCurrentInput(digit);
            state.setWaitingForOperand(false);

        } else {

            /*
             * Giới hạn độ dài input để tránh display quá dài.
             */
            if (current.length() >= 16) {
                return;
            }

            state.setCurrentInput(current + digit);
        }

        state.setResult(state.getCurrentInput());
    }


    // =========================================================
    // DECIMAL
    // =========================================================

    /**
     * Nhập dấu thập phân.
     */
    public void inputDecimal() {

        if (state.isError()) {
            state.reset();
        }

        /*
         * Ví dụ:
         *
         * 5 +
         * rồi nhấn "."
         *
         * → 0.
         */
        if (state.isWaitingForOperand()) {

            state.setCurrentInput("0.");
            state.setWaitingForOperand(false);

        } else {

            String current = state.getCurrentInput();

            /*
             * Không cho:
             *
             * 1.2.3
             */
            if (!current.contains(".")) {
                state.setCurrentInput(current + ".");
            }
        }

        state.setResult(state.getCurrentInput());
    }


    // =========================================================
    // OPERATOR
    // =========================================================

    /**
     * Chọn operator:
     *
     * +
     * −
     * ×
     * ÷
     */
    public void inputOperator(String operator) {

        if (!isValidOperator(operator)) {
            return;
        }

        if (state.isError()) {
            return;
        }

        double currentValue;

        try {

            currentValue =
                    Double.parseDouble(
                            state.getCurrentInput()
                    );

        } catch (NumberFormatException exception) {

            setErrorState();
            return;
        }


        /*
         * Hỗ trợ calculation liên tiếp:
         *
         * 2 + 3 + 4
         *
         * Khi user nhấn operator thứ hai,
         * engine tính pending operation trước.
         */
        if (state.getFirstOperand() != null
                && state.getOperator() != null
                && !state.isWaitingForOperand()) {

            Double intermediateResult =
                    performOperation(
                            state.getFirstOperand(),
                            currentValue,
                            state.getOperator()
                    );

            if (intermediateResult == null) {
                return;
            }

            currentValue = intermediateResult;

            String formatted =
                    NumberFormatter.format(
                            intermediateResult
                    );

            state.setCurrentInput(formatted);
            state.setResult(formatted);
        }


        state.setFirstOperand(currentValue);

        state.setOperator(operator);

        state.setExpression(
                NumberFormatter.format(currentValue)
                        + " "
                        + operator
        );

        /*
         * Digit tiếp theo phải thay currentInput,
         * không append vào operand cũ.
         */
        state.setWaitingForOperand(true);
    }


    // =========================================================
    // EQUALS
    // =========================================================

    /**
     * Thực hiện phép tính khi nhấn "=".
     */
    public void calculate() {

        if (state.isError()) {
            return;
        }

        /*
         * Không có operator đang chờ.
         *
         * Ví dụ user chỉ nhập:
         *
         * 25 =
         *
         * → giữ 25.
         */
        if (state.getFirstOperand() == null
                || state.getOperator() == null) {

            return;
        }


        /*
         * Nếu user vừa bấm operator rồi "=":
         *
         * 5 +
         * =
         *
         * không thực hiện calculation.
         */
        if (state.isWaitingForOperand()) {
            return;
        }


        double secondOperand;

        try {

            secondOperand =
                    Double.parseDouble(
                            state.getCurrentInput()
                    );

        } catch (NumberFormatException exception) {

            setErrorState();
            return;
        }


        double firstOperand =
                state.getFirstOperand();

        String operator =
                state.getOperator();


        Double calculatedResult =
                performOperation(
                        firstOperand,
                        secondOperand,
                        operator
                );

        if (calculatedResult == null) {
            return;
        }


        String formattedResult =
                NumberFormatter.format(
                        calculatedResult
                );


        /*
         * Expression cuối:
         *
         * 25 × 4
         */
        state.setExpression(

                NumberFormatter.format(firstOperand)

                        + " "

                        + operator

                        + " "

                        + NumberFormatter.format(secondOperand)
        );


        /*
         * Result:
         *
         * 100
         */
        state.setCurrentInput(formattedResult);

        state.setResult(formattedResult);


        /*
         * Calculation hiện tại kết thúc.
         */
        state.setFirstOperand(null);

        state.setOperator(null);


        /*
         * Sau "=":
         *
         * 25 × 4 = 100
         *
         * nếu nhấn 7
         *
         * → bắt đầu calculation mới với 7.
         *
         * Nếu nhấn +
         *
         * → dùng 100 làm firstOperand.
         */
        state.setWaitingForOperand(true);
    }


    // =========================================================
    // CLEAR
    // =========================================================

    /**
     * AC - reset toàn bộ calculation hiện tại.
     *
     * Không liên quan History.
     */
    public void clear() {
        state.reset();
    }


    // =========================================================
    // BACKSPACE
    // =========================================================

    /**
     * Xóa ký tự cuối cùng của current input.
     */
    public void backspace() {

        if (state.isError()) {

            state.reset();
            return;
        }


        /*
         * Nếu vừa chọn operator:
         *
         * 25 ×
         *
         * backspace không làm gì ở MVP.
         */
        if (state.isWaitingForOperand()) {
            return;
        }


        String current =
                state.getCurrentInput();


        if (current.length() <= 1) {

            state.setCurrentInput("0");

        } else {

            String updated =
                    current.substring(
                            0,
                            current.length() - 1
                    );

            /*
             * Trường hợp:
             *
             * -5
             *
             * xóa 5
             *
             * còn "-"
             *
             * phải chuyển về 0.
             */
            if ("-".equals(updated)
                    || updated.isEmpty()) {

                updated = "0";
            }

            state.setCurrentInput(updated);
        }


        state.setResult(
                state.getCurrentInput()
        );
    }


    // =========================================================
    // TOGGLE SIGN
    // =========================================================

    /**
     * Đổi dấu:
     *
     * 25  -> -25
     * -25 -> 25
     */
    public void toggleSign() {

        if (state.isError()) {
            state.reset();
        }


        /*
         * Nếu vừa chọn operator và chưa nhập operand mới,
         * bắt đầu operand mới từ 0.
         */
        if (state.isWaitingForOperand()) {

            state.setCurrentInput("0");
            state.setWaitingForOperand(false);
        }


        String current =
                state.getCurrentInput();


        /*
         * Không tạo "-0".
         */
        if ("0".equals(current)
                || "0.".equals(current)) {

            return;
        }


        if (current.startsWith("-")) {

            state.setCurrentInput(
                    current.substring(1)
            );

        } else {

            state.setCurrentInput(
                    "-" + current
            );
        }


        state.setResult(
                state.getCurrentInput()
        );
    }


    // =========================================================
    // PERCENT
    // =========================================================

    /**
     * MVP percent:
     *
     * x% = x / 100
     *
     * Ví dụ:
     *
     * 25%
     *
     * → 0.25
     */
    public void percent() {

        if (state.isError()) {
            state.reset();
        }

        if (state.isWaitingForOperand()) {
            return;
        }


        double value;

        try {

            value =
                    Double.parseDouble(
                            state.getCurrentInput()
                    );

        } catch (NumberFormatException exception) {

            setErrorState();
            return;
        }


        double percentValue =
                value / 100.0;


        String formatted =
                NumberFormatter.format(
                        percentValue
                );


        state.setExpression(
                NumberFormatter.format(value)
                        + "%"
        );

        state.setCurrentInput(formatted);

        state.setResult(formatted);
    }


    // =========================================================
    // CALCULATION CORE
    // =========================================================

    /**
     * Thực hiện phép toán thực tế.
     *
     * Return null nếu có lỗi.
     */
    private Double performOperation(
            double firstOperand,
            double secondOperand,
            String operator
    ) {

        double result;


        switch (operator) {

            case "+":
                result =
                        firstOperand
                                + secondOperand;
                break;


            case "-":
            case "−":
                result =
                        firstOperand
                                - secondOperand;
                break;


            case "*":
            case "×":
                result =
                        firstOperand
                                * secondOperand;
                break;


            case "/":
            case "÷":

                if (secondOperand == 0) {

                    setErrorState();

                    return null;
                }

                result =
                        firstOperand
                                / secondOperand;

                break;


            default:

                setErrorState();

                return null;
        }


        /*
         * Bảo vệ overflow/invalid result.
         */
        if (Double.isInfinite(result)
                || Double.isNaN(result)) {

            setErrorState();

            return null;
        }


        return result;
    }


    // =========================================================
    // VALIDATION
    // =========================================================

    private boolean isValidOperator(
            String operator
    ) {

        if (operator == null) {
            return false;
        }

        switch (operator) {

            case "+":
            case "-":
            case "−":
            case "*":
            case "×":
            case "/":
            case "÷":

                return true;

            default:

                return false;
        }
    }


    // =========================================================
    // ERROR
    // =========================================================

    /**
     * Đưa calculator vào trạng thái Error.
     */
    private void setErrorState() {

        state.setError(true);

        state.setExpression("");

        state.setCurrentInput("Error");

        state.setResult("Error");

        state.setFirstOperand(null);

        state.setOperator(null);

        state.setWaitingForOperand(false);
    }
}