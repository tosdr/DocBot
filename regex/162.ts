import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^(((?=.*non-refundable))|((?=.*not)((?=.*refundable))|(?=.*no)(?=.*refund)))", "i"),
    expressionDont: new RegExp("", "i"),
    caseID: 162,
    name: "The service has a no refund policy"
} as Regex;