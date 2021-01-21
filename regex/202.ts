import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^(((?=.*content)|(?=.*UGC)|(?=.*posting))((?=.*remove)|(?=.*refuse)|(?=.*delete))((?=.*without prior)((?=.*notice)|(?=.*notification))|((?=.*any reason)|(?=.*no reason)|(?=.*sole discretion))))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 202,
	name: "The service can delete specific content without reason and may do it without prior notice"
} as Regex;